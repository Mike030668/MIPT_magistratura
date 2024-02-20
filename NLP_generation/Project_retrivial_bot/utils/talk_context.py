
import torch
import numpy as np
import gc

def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    with torch.no_grad():
        for _ in range(3):
          torch.cuda.empty_cache()
          torch.cuda.ipc_collect()


def get_best_rand_reply(
    finetuned_ce,
    query: str,
    context: str,
    corpus: list[str],
    max_length,
    size_patch = 150,
    qty_rand_choose = 5,
    max_out_context = 200,

) -> None:

    dic_answ = dict()
    dic_answ["score"] = []
    dic_answ["answer"] = []

    conext_memory= query+"[SEP]"+context

    if len(corpus) < qty_rand_choose*max_out_context:
       qty_rand_choose = int(len(corpus))

    # так как база большая
    for i in range(qty_rand_choose):
        rand_patch_corpus = list(np.random.choice(corpus, size_patch))
        #print(len(rand_patch_corpus))

        queries = [conext_memory]* len(rand_patch_corpus)
        #print(len(queries))
        tokenized_texts = finetuned_ce.bert_tokenizer(
            queries,
            rand_patch_corpus,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(finetuned_ce.device)

        # Finetuned CrossEncoder model scoring
        with torch.no_grad():
            ce_scores = finetuned_ce(tokenized_texts['input_ids'],
                                     tokenized_texts['attention_mask']).squeeze(-1)
            ce_scores = torch.sigmoid(ce_scores)  # Apply sigmoid if needed

        # Process scores for finetuned model
        scores = ce_scores.cpu().numpy()
        scores_ix = np.argsort(scores)[::-1][0]
        dic_answ["score"].append(scores[scores_ix])
        dic_answ["answer"].append(rand_patch_corpus[scores_ix])

    id = np.argsort(dic_answ["score"])[::-1][0]# np.array(dic_answ["score"]).argmax()
    answer = dic_answ["answer"][id]
    conext_memory = answer+"[SEP]"+conext_memory
    flush_memory()
    return answer, conext_memory[:max_out_context], dic_answ["score"][id]
