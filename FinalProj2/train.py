import torch
import sacrebleu

def train_step(src, tgt, model, optimizer, config):
    model.train()
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    src_mask = (src != config.src_PAD_IDX).unsqueeze(1).unsqueeze(2)
    tgt_mask = model.create_look_ahead_mask(tgt_input.size(1)).to(config.device)
    tgt_padding_mask = (tgt_input != config.tgt_PAD_IDX).unsqueeze(1).unsqueeze(2)

    optimizer.zero_grad()
    logits, loss = model(src, tgt_input, src_mask, tgt_mask, tgt_padding_mask)
    loss.backward()
    optimizer.step()

    return loss.item()
import torch
import sacrebleu

def evaluate(model, dataloader, src_tokenizer, tgt_tokenizer, config):
    model.eval()
    total_loss = 0
    translations = []
    references = []

    with torch.no_grad():
        for batch in dataloader:
            src, tgt = batch
            src, tgt = src.to(config.device), tgt.to(config.device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = (src != config.src_PAD_IDX).unsqueeze(1).unsqueeze(2).to(config.device)
            tgt_mask = model.create_look_ahead_mask(tgt_input.size(1)).to(config.device)
            tgt_padding_mask = (tgt_input != config.tgt_PAD_IDX).unsqueeze(1).unsqueeze(2).to(config.device)

            logits, loss = model(src, tgt_input, src_mask, tgt_mask, tgt_padding_mask)
            total_loss += loss.item()

            # Generate translations
            generated_tokens = model.generate(src, max_new_tokens=config.max_seq_length)
            generated_texts = [tgt_tokenizer.decode(g.tolist(), skip_special_tokens=True) for g in generated_tokens]
            translations.extend(generated_texts)

            # Add reference translations
            ref_texts = [tgt_tokenizer.decode(t.tolist(), skip_special_tokens=True) for t in tgt_output]
            references.extend(ref_texts)

          

    # Compute BLEU score
    bleu = sacrebleu.corpus_bleu(translations, [references])
    bleu_score_percentage = bleu.score * 100
    print(f"Translation (sample): {generated_texts[:2]}")
    print(f"Reference (sample): {ref_texts[:2]}")
    avg_loss = total_loss / len(dataloader)
    model.train()
    return avg_loss, bleu_score_percentage
