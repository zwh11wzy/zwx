import json
import random
from datasets import load_dataset
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, wait

# 常量
API_KEY = "sk-8BZ60ZT58hPBmwvFn6AHaUjcmwH1lECIFefN1YZYP2vjYaS3"
BASE_URL = "https://api.closeai-asia.com/v1"  # 请确认此URL
MODELS = ["deepseek-reasoner", "gpt-3.5-turbo"]
BATCH_SIZE = 20
NUM_TEST_TRIPLES = 500


def load_data():
    dataset = load_dataset("KGraph/FB15k-237")
    train_dataset = dataset["train"]
    print(train_dataset)
    with open("./FB15K-237/entity2wikidata.json", "r") as f:
        entity_data = json.load(f)
    entity_to_name = {}
    for mid, data in entity_data.items():
        if isinstance(data, list) and data:
            data = data[0]  # 如果是列表，取第一个元素
        if isinstance(data, dict):
            wikidata_id = list(data.keys())[0]
            if "label" in data[wikidata_id]:
                entity_to_name[mid] = data[wikidata_id]["label"]
            else:
                entity_to_name[mid] = mid
        else:
            entity_to_name[mid] = mid
    return train_dataset, entity_to_name


def build_relation_to_tails(train_dataset):
    print("调试：打印前5个条目以检查格式")
    for i, item in enumerate(train_dataset[:5]):
        print(f"条目 {i}: {item}")
    relation_to_tails = {}
    train_triples = set()
    for item in train_dataset:
        if isinstance(item, dict) and all(key in item for key in ["head", "relation", "tail"]):
            h, r, t = item["head"], item["relation"], item["tail"]
            train_triples.add((h, r, t))
            if r not in relation_to_tails:
                relation_to_tails[r] = set()
            relation_to_tails[r].add(t)
        else:
            print(f"警告：条目 {item} 格式不正确，跳过")
    return relation_to_tails, train_triples


def generate_erroneous_triplets(train_dataset, relation_to_tails, train_triples, fraction=0.2):
    num_to_select = int(fraction * len(train_dataset))
    selected_indices = random.sample(range(len(train_dataset)), num_to_select)
    erroneous_triplets = []
    for idx in selected_indices:
        item = train_dataset[idx]
        h, r, t = item["head"], item["relation"], item["tail"]
        existing_tails = {t_ for (h_, r_, t_) in train_triples if h_ == h and r_ == r}
        candidates = relation_to_tails[r] - existing_tails
        if candidates:
            new_t = random.choice(list(candidates))
            erroneous_triplets.append({"head": h, "relation": r, "tail": new_t})
    return erroneous_triplets


def prepare_test_set(train_dataset, erroneous_triplets, num_samples=NUM_TEST_TRIPLES):
    all_triplets = [(item, "correct") for item in train_dataset] + [(item, "incorrect") for item in erroneous_triplets]
    test_triplets = random.sample(all_triplets, num_samples)
    return test_triplets


def map_to_names(test_triplets, entity_to_name):
    named_triplets = []
    for triplet, label in test_triplets:
        h_name = entity_to_name.get(triplet["head"], triplet["head"])
        t_name = entity_to_name.get(triplet["tail"], triplet["tail"])
        r = triplet["relation"]
        named_triplets.append((h_name, r, t_name, label))
    return named_triplets


def batch_triplets(named_triplets, batch_size=BATCH_SIZE):
    batches = [named_triplets[i : i + batch_size] for i in range(0, len(named_triplets), batch_size)]
    return batches


def construct_prompt(batch):
    prompt = "You are given a list of knowledge graph triples. For each triple, determine if it is correct, incorrect, or if you are uncertain. Respond with 'correct', 'incorrect', or 'uncertain' for each triple.\n\nTriples:\n"
    for i, (h, r, t, _) in enumerate(batch, 1):
        prompt += f"{i}. ({h}, {r}, {t})\n"
    prompt += "Please provide your judgments in order, one per line."
    return prompt


def process_batch(model, batch, client):
    prompt = construct_prompt(batch)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0,
    )
    judgments = response.choices[0].message.content.strip().split("\n")
    judgments = judgments[: len(batch)]
    while len(judgments) < len(batch):
        judgments.append("uncertain")
    return judgments


def compute_metrics(results, labels):
    tn = fn = tp = fp = uncertain = 0
    for judgment, label in zip(results, labels):
        judgment = judgment.lower().strip()
        if label == "incorrect":
            if judgment == "incorrect":
                tn += 1
            elif judgment == "correct":
                fn += 1
            else:
                uncertain += 1
        else:  # label == "correct"
            if judgment == "correct":
                tp += 1
            elif judgment == "incorrect":
                fp += 1
            else:
                uncertain += 1
    total = len(labels)
    correct = (tp + tn) / total * 100
    uncertain_rate = uncertain / total * 100
    error = (fp + fn) / total * 100
    return tn, fn, tp, fp, correct, uncertain_rate, error


def main():
    train_dataset, entity_to_name = load_data()
    relation_to_tails, train_triples = build_relation_to_tails(train_dataset)
    erroneous_triplets = generate_erroneous_triplets(train_dataset, relation_to_tails, train_triples)
    test_triplets = prepare_test_set(train_dataset, erroneous_triplets)
    named_triplets = map_to_names(test_triplets, entity_to_name)
    batches = batch_triplets(named_triplets)

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    for model in MODELS:
        print(f"Processing model: {model}")
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_batch, model, batch, client) for batch in batches]
            wait(futures)
            for future in futures:
                batch_results = future.result()
                results.extend(batch_results)

        labels = [label for _, label in test_triplets]
        tn, fn, tp, fp, correct, uncertain_rate, error = compute_metrics(results, labels)
        print(f"Model: {model}")
        print(f"TN: {tn}, FN: {fn}, TP: {tp}, FP: {fp}")
        print(f"Correct (%): {correct:.2f}")
        print(f"Not available (%): {uncertain_rate:.2f}")
        print(f"Error (%): {error:.2f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
