import csv
import json
import random
from datasets import load_dataset
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, wait

# 常量
API_KEY = "sk-8BZ60ZT58hPBmwvFn6AHaUjcmwH1lECIFefN1YZYP2vjYaS3"  # 请确认此API_KEY
BASE_URL = "https://api.openai-proxy.org/v1"  # 请确认此URL
MODELS = ["deepseek-chat", "gpt-3.5-turbo", "gpt-4"]
BATCH_SIZE = 20
NUM_TEST_TRIPLES = 500


def load_data():
    dataset = load_dataset("KGraph/FB15k-237")
    train_dataset = dataset["train"]
    print("Dataset column names:", train_dataset.column_names)
    for item in train_dataset.select(range(5)):
        print(item["text"])
    with open("./FB15k-237/entity2wikidata.json", "r") as f:
        entity_data = json.load(f)
    entity_to_name = {k: v.get("label", k) for k, v in entity_data.items()}
    return train_dataset, entity_to_name


def build_relation_to_tails(train_dataset):
    relation_to_tails = {}
    train_triples = set()
    train_triples_list = []
    for item in train_dataset:
        text = item["text"]
        parts = text.split("\t")
        if len(parts) == 3:
            h, r, t = parts
            train_triples.add((h, r, t))
            train_triples_list.append({"head": h, "relation": r, "tail": t})
            if r not in relation_to_tails:
                relation_to_tails[r] = set()
            relation_to_tails[r].add(t)
        else:
            print(f"警告：条目 {text} 格式不正确，跳过")
    return relation_to_tails, train_triples, train_triples_list


def generate_erroneous_triplets(train_triples_list, relation_to_tails, train_triples, fraction=0.2):
    num_to_select = int(fraction * len(train_triples_list))
    # selected_indices = random.sample(range(len(train_triples_list)), num_to_select)
    selected_indices = random.sample(range(len(train_triples_list)), num_to_select)
    print(f"选中的索引数量: {len(selected_indices)}")  # 输出选中的索引数量
    erroneous_triplets = []
    for idx in selected_indices:
        print(f"当前循环次数: {len(erroneous_triplets) + 1}")  # 表示这是第几次循环
        triplet = train_triples_list[idx]
        h, r, t = triplet["head"], triplet["relation"], triplet["tail"]
        existing_tails = {t_ for (h_, r_, t_) in train_triples if h_ == h and r_ == r}
        candidates = relation_to_tails[r] - existing_tails
        if candidates:
            new_t = random.choice(list(candidates))
            erroneous_triplets.append({"head": h, "relation": r, "tail": new_t})
        # if len(erroneous_triplets) == 10:
        #     break
    return erroneous_triplets


def prepare_test_set(train_triples_list, erroneous_triplets, num_samples=NUM_TEST_TRIPLES):
    all_triplets = [(triplet, "correct") for triplet in train_triples_list] + [
        (triplet, "incorrect") for triplet in erroneous_triplets
    ]
    test_triplets = random.sample(all_triplets, num_samples)

    # 将生成的三元组存到同级目录的新建的csv文件中
    with open("test_triplets.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["head", "relation", "tail", "label"])  # 写入表头
        for triplet, label in test_triplets:
            writer.writerow([triplet["head"], triplet["relation"], triplet["tail"], label])  # 写入数据行

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

    # 添加此处理逻辑
    cleaned_judgments = []
    for j in judgments:
        j = j.lower().strip()
        # 移除序号前缀
        if "." in j:
            j = j.split(".", 1)[1].strip()
        cleaned_judgments.append(j)

    while len(cleaned_judgments) < len(batch):
        cleaned_judgments.append("uncertain")
    return cleaned_judgments


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
    print("11\n")
    relation_to_tails, train_triples, train_triples_list = build_relation_to_tails(train_dataset)
    print("22\n")
    erroneous_triplets = generate_erroneous_triplets(train_triples_list, relation_to_tails, train_triples)
    print("33\n")
    test_triplets = prepare_test_set(train_triples_list, erroneous_triplets)
    print("44\n")
    named_triplets = map_to_names(test_triplets, entity_to_name)
    print("55\n")
    batches = batch_triplets(named_triplets)
    print("66\n")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    for epoch, model in enumerate(MODELS, start=1):
        print(f"Processing model: {model} (Epoch {epoch}/{len(MODELS)})")
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_batch, model, batch, client) for batch in batches]
            wait(futures)
            for i, future in enumerate(futures, start=1):
                batch_results = future.result()
                results.extend(batch_results)
                print(f"Batch {i}/{len(batches)} processed. Results: {batch_results}")

        labels = [label for _, label in test_triplets]
        tn, fn, tp, fp, correct, uncertain_rate, error = compute_metrics(results, labels)
        print(f"Model: {model}")
        print(f"TN: {tn}, FN: {fn}, TP: {tp}, FP: {fp}")
        print(f"Correct (%): {correct:.2f}")
        print(f"Not available (%): {uncertain_rate:.2f}")
        print(f"Error (%): {error:.2f}")
        print("-" * 50)
        with open("model_metrics.csv", mode="a+", newline="") as file:
            writer = csv.writer(file)
            # 写入标题行，如果文件为空则写入标题
            file.seek(0, 2)  # 移动到文件末尾
            if file.tell() == 0:  # 检查文件是否为空
                writer.writerow(["Model", "TN", "FN", "TP", "FP", "Correct (%)", "Not available (%)", "Error (%)"])
            writer.writerow([model, tn, fn, tp, fp, f"{correct:.2f}", f"{uncertain_rate:.2f}", f"{error:.2f}"])


if __name__ == "__main__":
    main()
