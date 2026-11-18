import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data, get_tokenizer, sql_decode
from utils import compute_metrics, save_queries_and_records, set_random_seeds

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=25,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=5,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=32)

    # Generation
    parser.add_argument('--num_beams', type=int, default=4, help="Beam size for generation (1 = greedy)")
    parser.add_argument('--max_gen_len', type=int, default=512, help="Max number of generated tokens")
    parser.add_argument('--train_eval_beams', type=int, default=1,
                        help="Beam size for the per-epoch dev evaluation during training (1 = greedy, faster)")
    parser.add_argument('--bf16', type=int, default=1, help="Use bfloat16 autocast on GPU")
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path,
                                                                         num_beams=args.train_eval_beams)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16(args)):
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad].float(), decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens
        
def use_bf16(args):
    return bool(args.bf16) and DEVICE.type == 'cuda'

def generate_sql(args, model, encoder_input, encoder_mask, initial_decoder_inputs, tokenizer, num_beams=None):
    '''Generate SQL strings (decoded back to real SQL) for one batch.'''
    num_beams = args.num_beams if num_beams is None else num_beams
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16(args)):
        out = model.generate(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=initial_decoder_inputs,
            max_new_tokens=args.max_gen_len,
            num_beams=num_beams,
            early_stopping=True,
        )
    texts = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [sql_decode(t) for t in texts]

def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path, num_beams=None):
    '''
    Evaluation loop used during training: dev loss (teacher forcing), then generation with the
    model, execution of the generated queries and the three metrics (SQL EM, record EM, record F1)
    plus the rate of generated queries that raised SQL errors.
    '''
    model.eval()
    tokenizer = get_tokenizer()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_tokens = 0, 0
    generated = []

    for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_inputs in tqdm(dev_loader):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)
        initial_decoder_inputs = initial_decoder_inputs.to(DEVICE)

        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16(args)):
            logits = model(input_ids=encoder_input, attention_mask=encoder_mask,
                           decoder_input_ids=decoder_input)['logits']
        with torch.no_grad():
            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad].float(), decoder_targets[non_pad])
            n = torch.sum(non_pad).item()
            total_loss += loss.item() * n
            total_tokens += n

        generated.extend(generate_sql(args, model, encoder_input, encoder_mask, initial_decoder_inputs,
                                      tokenizer, num_beams=num_beams))

    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    save_queries_and_records(generated, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(gt_sql_pth, model_sql_path,
                                                               gt_record_path, model_record_path)
    error_rate = sum(1 for m in error_msgs if m != "") / len(error_msgs)
    return total_loss / total_tokens, record_f1, record_em, sql_em, error_rate

def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    Generate SQL for the test set (no ground truth), then save the queries and their records.
    '''
    model.eval()
    tokenizer = get_tokenizer()
    generated = []
    for encoder_input, encoder_mask, initial_decoder_inputs in tqdm(test_loader):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        initial_decoder_inputs = initial_decoder_inputs.to(DEVICE)
        generated.extend(generate_sql(args, model, encoder_input, encoder_mask, initial_decoder_inputs, tokenizer))

    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    save_queries_and_records(generated, model_sql_path, model_record_path)

def main():
    # Get key arguments
    args = get_args()
    set_random_seeds(args.seed)
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
