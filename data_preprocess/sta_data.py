import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

base = ['train']
matplotlib.rc("font", family='Times New Roman')

code_num = {}
code_num_train = {}
code_num_dev = {}
code_num_test = {}

colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']  # 自定义颜色

syndromes_text = {}
with open('./syndrome_knowledge.json', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        name = data['Name']
        text = data['Definition'] + '[SEP]' + data['Typical_performance'] + '[SEP]' + data['Common_isease']
        syndromes_text[name] = text

dayu_500 = 0
num_dict = {}

for basei in base:
    with open('./clear_{}.json'.format(basei), 'r', encoding='utf-8') as file:
        num = 0
        text_len = 0
        zy = []
        code_text = 0
        code_num_base = {}
        for line in tqdm(file):
            content = json.loads(line)
            text = content['chief_complaint'] + '[SEP]' + content['description'] + '[SEP]' + content['detection']
            num_dict[num] = len(text)
            if len(text) > 500 :
                dayu_500 = dayu_500 + 1
            text_len = text_len + len(text)
            num = num + 1
            zy_icd_i = content['norm_syndrome']
            zy.append(zy_icd_i)

            if zy_icd_i in code_num:
                code_num[zy_icd_i] = code_num[zy_icd_i] + 1
            else:
                code_num[zy_icd_i] = 1

            if zy_icd_i in code_num_base:
                code_num_base[zy_icd_i] = code_num_base[zy_icd_i] + 1
            else:
                code_num_base[zy_icd_i] = 1

            if basei == 'train':
                code_num_train = code_num_base
            elif basei == 'dev':
                code_num_dev = code_num_base
            elif basei == 'test':
                code_num_test = code_num_base

            code_text = code_text + len(syndromes_text[zy_icd_i])

        sorted_items_base = sorted(code_num_base.items(), key=lambda x: x[1], reverse=True)
        sorted_dict_base = dict(sorted_items_base)
        print("sorted_dict_base",sorted_dict_base)

        num_dict_base = sorted(num_dict.items(), key=lambda x: x[1], reverse=True)
        num_dict_base = dict(num_dict_base)
        print("num_dict_base", num_dict_base)

        text_len_ave = text_len / num
        print(len(list(set(zy))))
        print(text_len_ave)
        print(code_text / num)

sorted_items = sorted(code_num.items(), key=lambda x: x[1], reverse=True)
sorted_dict = dict(sorted_items)
print(sorted_dict)
print(num)

print('dayu500',dayu_500)

# 提取字典的值
values_dict1 = list(sorted_dict_base.values())
values_dict2 = list(code_num_dev.values())
values_dict3 = list(code_num_test.values())

num_dict1 = list(num_dict_base.values())
values_dict1 =num_dict1

# 创建图形对象
fig, ax = plt.subplots(figsize=(8, 6))

# 去掉上边和右边的边框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ii_500 = 0
ii_300 = 0
ii_50 = 0

for i,xx in enumerate(values_dict1):
    if xx < 500 and ii_500 == 0:
        ii_500 = i
    if xx < 200 and ii_300 == 0:
        ii_300 = i
    if xx < 10 and ii_50 == 0:
        ii_50 = i
    else:
        continue


# 将x轴的起点与y轴的0刻度对齐
ax.spines['bottom'].set_position(('zero'))

# 绘制与y轴平行的线
# 绘制竖线
# print(ii)
# print(values_dict1)
# print(len(values_dict1))
# 绘制竖线
# 绘制竖线
x_values = [ii_500]  # 替换为你的竖线x坐标列表
print(x_values)
for x in x_values:
    plt.axvline(x=x, color='gray', linestyle='dashed', linewidth=1)

# 修改刻度线颜色和粗细
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['xtick.major.width'] = 4
plt.rcParams['ytick.major.width'] = 4

# 绘制折线图
plt.plot(values_dict1, color=colors[0], linewidth=1.8)
# plt.plot(values_dict2, color=colors[2], label='Dev', linewidth=2.5)
# plt.plot(values_dict3, color=colors[4], label='Test', linewidth=2.5)

# 设置y轴范围
ax.set_ylim(bottom=0)
plt.xlim(left=0)

plt.xlabel('Sorted Clinical Text ID')
plt.ylabel('Lengths')

# 调整图例字体大小
plt.legend(fontsize='large')

plt.savefig('./data_num.pdf')

plt.show()