import json

d1=json.load(open('/media/disk2/daic/blur_data/tree_node/mTreeNodeIndex_addVal4w_refined.json'))
tlist=d1['train'].keys()
slist=d1['test'].keys()
vlist=d1['val'].keys()
v4list=d1['val4w'].keys()
l1=json.load(open('/media/disk2/daic/blur_data/tree_node/mTreeNodeIndex_addVal4w_refined3.json'))

# print(d1['train'][0][0])
# print(l1['train'][0])
list1=[[],[],[],[],[],[],[]]
list0=[[],[],[],[],[],[],[]]
def add_set(x,set):
    if x not in set:
        set.append(x)

for x in range(len(tlist)):
    for y in range(5):
        for z in range(7):
            add_set(d1['train'][tlist[x]][y][z],list0[z])
for x in range(len(slist)):
    for y in range(5):
        for z in range(7):
            add_set(d1['test'][slist[x]][y][z],list0[z])
for x in range(len(vlist)):
    for y in range(5):
        for z in range(7):
            add_set(d1['val'][vlist[x]][y][z],list0[z])
for x in range(len(v4list)):
    for y in range(5):
        for z in range(7):
            add_set(d1['val4w'][v4list[x]][y][z],list0[z])

for x in range(len(l1['train'])):
    for z in range(7):
        add_set(l1['train'][x][z],list1[z])
for x in range(len(l1['val'])):
    for z in range(7):
        add_set(l1['val'][x][z],list1[z])
for x in range(len(l1['test'])):
    for z in range(7):
        add_set(l1['test'][x][z],list1[z])
for z in range(7):
    print('0',len(list0[z]),z)
    print('1',len(list1[z]),z)
enti=[0,2,4,6]
rel=[1,3,5]
en=[]#(145, 799)
re=[]
for x in enti:
    for y in range(len(list1[x])):
        if list1[x][y] not in en:
            en.append(list1[x][y])
for x in rel:
    for y in range(len(list1[x])):
        if list1[x][y] not in re:
            re.append(list1[x][y])
print(list1[5])
print(len(re),len(en))
print(max(re),max(en))