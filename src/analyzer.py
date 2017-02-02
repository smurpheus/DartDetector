import os
import csv
videos = []
for fn in os.listdir('.'):
    if os.path.isfile(fn) and fn.split('.')[-1] == "avi":
        videos.append(fn)
print videos
print len(videos)
def isbool(val):
    if val == "True":
        return True
    else:
        return False
def wasCorrect(list1, list2):
    if len(list1) != len(list2):
        return False
    correct = []
    wrong = []
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            correct.append(i)
        else:
            wrong.append(i)
    return correct, wrong
def calcPercentage(est, real):
    cor, wrong = wasCorrect(est, real)
    inneigh1 = []
    for each in wrong:
        if str(est[each]) in neighbours()[str(real[each])]:
            inneigh1.append(each)
    perc = float(len(cor)) / len(real)
    wperc = float(len(wrong)) / len(real)
    percneigh = float(len(inneigh1))/len(wrong)
    posbile = wperc * percneigh
    return perc, percneigh, posbile

def calc(data):
    if len(data) == 0:
        return False
    tip1 = [x[0] for x in data]
    tip2 = [x[1] for x in data]
    tip3 = [x[2] for x in data]
    tip4 = [x[3] for x in data]
    tip5 = [x[4] for x in data]
    real = [x[5] for x in data]
    perc1, percneigh1, posbile1 = calcPercentage(tip1, real)
    perc2, percneigh2, posbile2 = calcPercentage(tip2, real)
    perc3, percneigh3, posbile3 = calcPercentage(tip3, real)
    perc4, percneigh4, posbile4 = calcPercentage(tip4, real)
    perc5, percneigh5, posbile5 = calcPercentage(tip5, real)

    print "Percplain", perc1, perc2, perc3, perc4, perc5
    print "Perc Neighbour", percneigh1,percneigh2, percneigh3,percneigh4,percneigh5
    print "Posible", posbile1, posbile2, posbile3, posbile4, posbile5
    tip1 = [x[0] for x in data if not isbool(x[6])]
    tip2 = [x[1] for x in data if not isbool(x[6])]
    tip3 = [x[2] for x in data if not isbool(x[6])]
    tip4 = [x[3] for x in data if not isbool(x[6])]
    tip5 = [x[4] for x in data if not isbool(x[6])]
    real = [x[5] for x in data if not isbool(x[6])]
    if len(real) != 0:
        perc1, percneigh1, posbile1 = calcPercentage(tip1, real)
        perc2, percneigh2, posbile2 = calcPercentage(tip2, real)
        perc3, percneigh3, posbile3 = calcPercentage(tip3, real)
        perc4, percneigh4, posbile4 = calcPercentage(tip4, real)
        perc5, percneigh5, posbile5 = calcPercentage(tip5, real)
        print "Percplain", perc1, perc2, perc3, perc4, perc5
        print "Perc Neighbour", percneigh1, percneigh2, percneigh3, percneigh4, percneigh5
        print "Posible", posbile1, posbile2, posbile3, posbile4, posbile5
    else:
        print "-", "-", "-", "-", "-"
    tip1 = [x[0] for x in data if  isbool(x[6])]
    tip2 = [x[1] for x in data if  isbool(x[6])]
    tip3 = [x[2] for x in data if  isbool(x[6])]
    tip4 = [x[3] for x in data if  isbool(x[6])]
    tip5 = [x[4] for x in data if  isbool(x[6])]
    real = [x[5] for x in data if  isbool(x[6])]
    print float(len(real))/float(len(data))
    # print [x[6] for x in data]
    if len(real) != 0:
        perc1, percneigh1, posbile1 = calcPercentage(tip1, real)
        perc2, percneigh2, posbile2 = calcPercentage(tip2, real)
        perc3, percneigh3, posbile3 = calcPercentage(tip3, real)
        perc4, percneigh4, posbile4 = calcPercentage(tip4, real)
        perc5, percneigh5, posbile5 = calcPercentage(tip5, real)
        print "Percplain", perc1, perc2, perc3, perc4, perc5
        print "Perc Neighbour", percneigh1, percneigh2, percneigh3, percneigh4, percneigh5
        print "Posible", posbile1, posbile2, posbile3, posbile4, posbile5
    else:
        print "-", "-", "-", "-", "-"
alldata = []
for vid in videos:
    datavid = []
    for fn in os.listdir('.'):
        if os.path.isfile(fn) and fn.split('.')[-1] == "csv" and vid.split('.')[0] in fn:
            datavid.append(fn)
    print vid, sorted(datavid)
    alldata += datavid
data = []
header = None
# print alldata
for fn in alldata:
    # print fn
    with open(fn, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        datas = list(spamreader)
        if header is None:
            header = datas[0]
        print fn
        for row in datas[1:]:
            data.append(row)
# with open("alldata.csv","w") as output:
#     spamwriter = csv.writer(output, delimiter=',')
#     spamwriter.writerow(header)
#     for entry in data:
#         spamwriter.writerow(entry)
#
def neighbours():
    fields_in_order = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    neighbours = {}
    for each in fields_in_order:
        i = fields_in_order.index(each)
        n1 = fields_in_order[i - 1]
        if i + 1 > len(fields_in_order) - 1:
            n2 = fields_in_order[0]
        else:
            n2 = fields_in_order[i + 1]
        neighbours["%s"%each] = [str(n1), str(n2), "D%s"%each, "T%s"%each, "D%s"%n1, "T%s"%n1,"D%s"%n2, "T%s"%n2, "%s"%25]
        neighbours["D%s"%each] = ["D%s"%n1, "D%s"%n2, "%s"%n1, "%s"%n2, "%s"%0, "%s"%each]
        neighbours["T%s"%each] = ["T%s"%n1, "T%s"%n2, "%s"%n1, "%s"%n2, "%s"%each]
    neighbours["%s"%25] = ["%s"%x for x in fields_in_order]
    neighbours["%s"%25].append("%s"%50)
    neighbours["%s"%50] = ["%s"%25]
    neighbours["0"] = ["D%s"%x for x in fields_in_order]
    return neighbours
calc(data)



