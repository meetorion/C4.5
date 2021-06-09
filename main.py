import linecache
import math
import sys

sys.setrecursionlimit(100000)  # 例如这里设置为十万
from graphviz import Digraph

g = Digraph('G', filename='c4.5.gv')
# 声明一个全局字典来存储编号(列号)和节点的映射关系cnt => row
cntToNode = {}
# rowToName = {'0': 'Outlook', '1': 'Temper.', '2': 'Humidity', '3': 'Windy'}
# nameToRow = {'Outlook': '0', 'Temper.': '1', 'Humidity': '2', 'Windy': '3'}
rowToName = {}
nameToRow = {}
edgeVal = {}  # 存　u=>v 对应的边的值
nodeVal = {}  # 存节点u对应的值
# 一个节点需要存储的信息有它的邻接节点的编号列表
cnt = 0
dict = {}


# 测试通过
def read(filename):
    dataSet = []
    ans = []
    with open(filename, 'r+') as file:
        for line in file:
            tmp = list(line.split(' '))
            dataSet.append(tmp)
    for line in dataSet:
        val = line[-1][0]
        line = line[:-1]
        line.append(str(val))
        ans.append(line)

    return ans


def create(dataSet):
    uniqueLabel = set([example[-1] for example in dataSet])
    if len(uniqueLabel) == 1:
        return None
    entropy = cal_entropy(dataSet)
    # 选出信息增益最大的特征　输入：数据集　输出特征所在列列号
    row = select(dataSet)
    sum = 0

    global cnt
    cnt = cnt + 1
    ans = cnt

    global cntToNode
    cntToNode[cnt] = row

    global g
    g.node(str(cnt), label=rowToName[str(cntToNode[cnt])])
    nodeVal[cnt] = rowToName[str(cntToNode[cnt])]

    # 根据列号分出子集并递归创建子树
    # row = 2
    uniqueValue = set([example[row] for example in dataSet])
    if len(uniqueValue) <= 1:
        return None

    edgeValue = None
    for value in uniqueValue:
        subDataSet = splitDataSet(dataSet, value, row)
        tmp = create(subDataSet)
        # 边和节点怎么存储，怎么连接呢？用字典
        if tmp == None:
            cnt = cnt + 1
            tmp = cnt
            label = subDataSet[0][-1]
            g.node(str(cnt), label='标签为：' + str(label))
            nodeVal[cnt] = str(label)

        else:

            g.node(str(tmp), label=rowToName[str(cntToNode[tmp])])
            nodeVal[tmp] = rowToName[str(cntToNode[tmp])]
        add_edge(ans, tmp, value)
    return ans
    # 测试
    # print(uniqueValue)


# 计算数据集的熵
def cal_entropy(dataSet):
    labelCounts = {}
    labelSum = len(dataSet)
    ans = 0
    for line in dataSet:
        label = line[-1]
        if label not in labelCounts:
            labelCounts[label] = 1
        else:
            labelCounts[label] += 1
    for label in labelCounts:
        pro = labelCounts[label] / labelSum
        ans -= pro * math.log(pro, 2)
    # print(ans)
    return ans

def get_unique_set(dataSet, index):
    return set([line[index] for line in dataSet])

def get_info(dataSet, index):

    uniqueValue = get_unique_set(dataSet, index)
    #统计每个类别的数量
    dict = {}
    tag = list(line[index] for line in dataSet)
    for val in tag:
        count = dict.get(val,0)
        dict[val] = count + 1

    #计算info
    info = 0
    sum = len(tag) #总标签个数
    for val in dict:
        pro = dict[val] / sum #val标签占整个标签的比例
        info -= pro * math.log(pro,2)
    return info


def select(dataSet):
    # 每个特征划分
    best = 1000000
    ans = -1
    attrSum = len(dataSet[0]) - 1
    attrCounts = {}
    info = get_info(dataSet, -1)  # 由标签类别获得的信息
    for i in range(attrSum):
        uniqueValue = set([example[i] for example in dataSet])

        curEntropy = 0 #熵
        gain = 0 #熵增益
        gain_ratio = 0 #熵增益率
        for value in uniqueValue:
            sum = 0 #每个属性的个数
            for line in dataSet:
                tmp = line[i]
                if value == tmp:
                    sum += 1

            subDataSet = splitDataSet(dataSet, value, i)
            pro = sum / len(dataSet)
            curEntropy += pro * cal_entropy(subDataSet) #熵最小准则


        #计算熵增益
        gain = info - curEntropy
        #计算划分后的信息
        split_info = get_info(dataSet, i)
        if split_info == 0:
            continue
        #计算熵增益率
        gain_ratio = gain / split_info
        gain_ratio = -gain_ratio
        # print(curEntropy)
        # if curEntropy < best:
        #     best = curEntropy
        #     ans = i
        #用熵增益率来更新
        if gain_ratio < best:
            best = gain_ratio
            ans = i

    return ans


# 测试通过
def splitDataSet(dataSet, value, row):
    ans = []
    for line in dataSet:
        if line[row] == value:
            # reducedLine = line[:row]
            # reducedLine.extend(line[row+1:])
            # ans.append(reducedLine)
            ans.append(line)
    return ans


def add_edge(u, v, value):  # u ==> v 并且在边上写value
    if u not in dict:
        dict[u] = set()
        dict[u].add(v)
    else:
        dict[u].add(v)
    # global g
    g.edge(str(u), str(v), label=str(value))
    edgeVal[str(u) + str(v)] = str(value)


def test():
    dataSet = read("dna.data")
    init(dataSet)
    root = create(dataSet)
    # g.view()
    exam(root, read("dna.test"))

def init(dataSet):
    numAttr = len(dataSet[0]) - 1
    for i in range(numAttr):
        rowToName[str(i)] = str(i)
        nameToRow[str(i)] = str(i)

def exam(root, dataSet):  # 输入：决策树的根节点编号，测试集　输出：错误率
    # num = len(dataSet[0])
    sum = 0
    num = 0
    for line in dataSet:
        num += 1
        if not_ok(line, root):
            sum += 1
    faultPro = sum / num
    print("错误率为：", faultPro)
    return faultPro


def not_ok(line, root):
    i = 0
    row = int(nameToRow[nodeVal[root]])
    while i < len(line):
        u = root
        set = dict.get(u,{})
        if len(set) == 0:
            label = nodeVal[u]
            if line[-1] == label:
                return False
            else:
                return True
        for v in dict[u]:
            tmp = str(u)+str(v)
            other = line[row]
            val = edgeVal[tmp]

            if val == line[row]:
                root = v
                i += 1

                attr = nodeVal[v]
                if isLabel(attr):
                    return attr != line[-1]
                row = int(nameToRow[nodeVal[root]])
                break
    return True


def isLabel(attr):
    if attr == '1' or attr == '0':
        return True
    return False


if __name__ == '__main__':
    test()
    # solve()


