import json
import tqdm

import nltk
from nltk.corpus import wordnet as wn

from collections import defaultdict
from collections import Counter

import plotly.graph_objects as go
import plotly.io as pio


# NLTK 데이터 다운로드 (최초 실행 시 필요)
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('words')


# JSON 파일 경로 설정
json_file_path = "generated_data_single_120000.json"  # 파일 경로를 JSON 파일로 교체

# JSON 데이터 로드
with open(json_file_path, 'r') as file:
    data = json.load(file)
sentences = [ s["sentence"]for s in data ]


# 문장에서 동사와 직접목적어 추출
def extract_verbs_and_objects(data, sentences, number_of_verbs=20, number_of_outer=5):
    verb_object_map = {}

    # Step 1: extract verb & direct_object in sentence
    for idx, sentence in enumerate(tqdm.tqdm(sentences)):
        tokens = nltk.word_tokenize(sentence)  # 문장 토큰화
        pos_tags = nltk.pos_tag(tokens)  # 품사 태깅
        grammar = "VP: {<VB.*><DT>?<NN.*>}"  # 동사-명사 구조를 위한 문법 규칙
        chunk_parser = nltk.RegexpParser(grammar)   # 구문 분석기
        tree = chunk_parser.parse(pos_tags)  # 구문 트리 생성
        
        once = False
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'VP'):  # VP (Verb Phrase)만 필터링
            if len(subtree) >= 2:  # 동사와 명사 확인
                verb = subtree[0][0]  # 첫 번째 항목은 동사
                obj = subtree[1][0]  # 두 번째 항목은 직접목적어

                # TODO
                if obj == "the" or obj == "an" or obj == "a" or obj == "others" \
                    or obj == "any" or obj == "anyone":

                    data[idx]['verb'], data[idx]['direct_object'] = None, None
                    continue

                if verb not in verb_object_map:
                    verb_object_map[verb] = []
                verb_object_map[verb].append(obj)  # 목적어 추가
                
                if ~once:
                    once = True
                else:
                    print ("check verb , object pairs")
                    from IPython import embed; embed()
                data[idx]['verb'], data[idx]['direct_object'] = verb, obj

            else:
                data[idx]['verb'], data[idx]['direct_object'] = None, None

    
    # Step 2: 동사별 빈도수 계산
    verb_data = []
    for verb, objects in verb_object_map.items():
        object_counts = Counter(objects)  # 목적어 빈도수 계산
        verb_data.append((verb, len(objects), object_counts.most_common()))

    verb_data = verb_data[:number_of_verbs]
    verb_data = [(verb_object_pair[0], verb_object_pair[1], verb_object_pair[2][:number_of_outer]) for verb_object_pair in verb_data]
    return data, verb_data

# 동사 빈도수 출력
data, result = extract_verbs_and_objects(data, sentences)
print(result)

# JSON 파일로 저장
output_file_path = "verb_obj_pair_termfrequency.json"
with open(output_file_path, 'w') as output_file:
    json.dump(result, output_file, indent=4)

output_data_path = f"extract_verb_object_{json_file_path}"
with open(output_data_path, 'w') as output_file2:
    json.dump(data, output_file2, indent=4)
    
print(f"결과가 {output_file_path} 파일에 저장되었습니다.")






# 각 동사에 연결된 명사를 매핑
verb_noun_map = defaultdict(list)

for verb, _, noun_freqs in result:
    for noun, _ in noun_freqs:
        verb_noun_map[verb].append(noun)

# Plotly Pie Chart 데이터 준비
labels, parents, values, ids = [], [], [], []


# 차트를 이미지와 HTML 파일로 저장
output_image_path, output_html_path = "sample_chart.png", "sample_chart.html"
from IPython import embed; embed()

# for verb, nouns in verb_noun_map.items():
#     labels.append(verb)
#     parents.append("")  # Root 노드
#     for noun in nouns:
#         labels.append(noun)
#         parents.append(verb)

conversations_v, people_v, someone_v, strangers_v, everyone_v, music_v, games_v = 0, 0, 0, 0, 0, 0, 0

for verb, verb_freqs, nouns_pairs in result:
    labels.append(verb)
    parents.append("")  # Root 노드
    values.append(verb_freqs)
    ids.append(f"{verb}_root")  # 고유 ID 추가
    
    total_freqs = 0
    for noun, noun_freqs in nouns_pairs:
        total_freqs += noun_freqs
    for noun, noun_freqs in nouns_pairs:
        if noun == "conversations":
            if conversations_v==0:
                labels.append(f"{noun}")
            else:
                labels.append(f"{noun} (v{conversations_v})")
            conversations_v += 1
        elif noun == "people":
            if people_v==0:
                labels.append(f"{noun} (v{people_v})")
            else: 
                labels.append(f"{noun}")
            people_v += 1
        elif noun == "someone":
            if someone_v==0:
                labels.append(f"{noun}")
            else:
                labels.append(f"{noun} (v{someone_v})")
            someone_v += 1
        elif noun == "strangers":
            if strangers_v==0:
                labels.append(f"{noun}")
            else:
                labels.append(f"{noun} (v{strangers_v})")
            strangers_v += 1
        elif noun == "everyone":
            if everyone_v==0:
                labels.append(f"{noun}")
            else:
                labels.append(f"{noun} (v{everyone_v})")
            everyone_v += 1
        elif noun == "music":
            if music_v==0:
                labels.append(f"{noun}")
            else:
                labels.append(f"{noun} (v{music_v})")
            music_v += 1
        elif noun == "games":
            if games_v==0:
                labels.append(f"{noun}")
            else:
                labels.append(f"{noun} (v{games_v})")
            games_v += 1
        else:
            labels.append(f"{noun}")
        # labels.append(f"{noun}")
        parents.append(verb)
        # values.append( noun_freqs )
        values.append( verb_freqs * (noun_freqs / total_freqs) )
        ids.append(f"{verb}_{noun}")  # 고유 ID 추가


# TODO: Debugging
for element in result:
    if element[0] == "hold" or element[0] == "playing" or element[0] == "assume" or element[0] == "invite" \
        or element[0] == "persuade" or element[0] == "harm" or element[0] == "watching":
        print (element)

# TODO: Debugging
for label, parent, value, id in zip(labels, parents, values, ids):
    if label == "the" or label=="other":
        print (label , parent, value, id)



# 중복 확인 및 디버깅
if len(ids) != len(set(ids)):
    print("Duplicate IDs detected:", [id for id in ids if ids.count(id) > 1])
assert len(labels) == len(parents) == len(values) == len(ids)


# Plotly Sunburst Chart 생성
fig = go.Figure(go.Sunburst(
    labels=labels,
    parents=parents,
    values=values,  # 각 노드의 크기를 정의
    # ids=ids,        # 고유 ID 추가
    branchvalues="total"
))

# 차트 디자인 설정
fig.update_layout(margin=dict(t=30, l=30, r=30, b=30))

# 이미지 파일 저장 (PNG)
pio.write_image(fig, output_image_path, format="png", width=800, height=800)

# HTML 파일 저장
pio.write_html(fig, output_html_path)

print(f"Chart saved as {output_image_path} and {output_html_path}")
