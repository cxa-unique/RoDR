import sys
from scipy import stats


def calculate_mrr_and_recall(qrels_file_path, output_results_path):

    qrel = {}
    with open(qrels_file_path, 'r', encoding='utf8') as f:
        # tsvreader = csv.reader(f, delimiter="\t")
        # for [topicid, _, docid, rel] in tsvreader:
        for line in f:
            topicid, _, docid, rel = line.strip().split()
            if rel == "0":
                continue
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]

    rankings = {}
    with open(output_results_path, 'r') as file:

        for line in file:
            [qid, _, docid, rank, _, _] = line.split()

            if qid not in rankings:
                rankings[qid] = []

            rankings[qid].append([docid, int(rank)])

    query_num = 0
    mrr_ten_score = 0.0
    recall_score = 0.0
    qid_mrr = {}
    qid_recall = {}

    for query in rankings:
        # if query not in qrel:
        #     continue
        assert query in qrel, 'query {} not in qrels!'.format(query)

        if query not in qid_mrr:
            qid_mrr[query] = {}
        if query not in qid_recall:
            qid_recall[query] = {}

        mrr_10_perquery = 0.

        query_num += 1
        for [doc, rank] in rankings[query][:10]:
            if doc in qrel[query]:
                mrr_ten_score += 1.0 / rank
                mrr_10_perquery = 1.0 / rank
                break

        hit_num = 0
        for [doc, rank] in rankings[query]:
            if doc in qrel[query]:
                hit_num += 1
        recall_score += hit_num / len(qrel[query])
        recall_perquery = hit_num / len(qrel[query])

        qid_recall[query]['recall'] = recall_perquery
        qid_mrr[query]['mrr_10'] = mrr_10_perquery

    mrr_ten = mrr_ten_score / query_num
    recall = recall_score / query_num

    return mrr_ten, qid_mrr, recall, qid_recall, query_num


def tt_test(qrels, res1, res2):
    mrr_ten1, mrr_dict1, recall_1, recall_dict_1, q_num1 = calculate_mrr_and_recall(qrels, res1)
    mrr_ten2, mrr_dict2, recall_2, recall_dict_2, q_num2 = calculate_mrr_and_recall(qrels, res2)

    print('res1: [mrr_10: {0}] [recall: {1}] [Q_num: {2}]'.format(str(mrr_ten1), str(recall_1), str(q_num1)))
    print('res2: [mrr_10: {0}] [recall: {1}] [Q_num: {2}]'.format(str(mrr_ten2), str(recall_2), str(q_num2)))

    mrr10_list1 = []
    mrr10_list2 = []
    for qid in mrr_dict1.keys():
        mrr10_list1.append(mrr_dict1[qid]['mrr_10'])
        mrr10_list2.append(mrr_dict2[qid]['mrr_10'])
    mrr10_p_value = stats.ttest_rel(mrr10_list1, mrr10_list2)[1]

    recall_list1 = []
    recall_list2 = []
    for qid in recall_dict_1.keys():
        recall_list1.append(recall_dict_1[qid]['recall'])
        recall_list2.append(recall_dict_2[qid]['recall'])
    recall_p_value = stats.ttest_rel(recall_list1, recall_list2)[1]
    print('tt_test_p_value: [mrr_10: {}] [recall: {}]'.format(str(mrr10_p_value), str(recall_p_value)))


def fusion_tt_test(qrels, setting_1, setting_2):
    print('#################################################################')
    print(setting_1, '-- vs --', setting_2)
    print('## Original Res ##')
    res1 = '{}/queries.dev.small.q32.p128.top1k.run.txt'.format(setting_1)
    res2 = '{}/queries.dev.small.q32.p128.top1k.run.txt'.format(setting_2)
    tt_test(qrels, res1, res2)

    VARIATION_TYPES=['MisSpell', 'ExtraPunc', 'BackTrans', 'SwapSyn_Glove', 'SwapSyn_WNet', 'TransTense', 'NoStopword', 'SwapWords']

    mrr10_list1 = []
    mrr10_list2 = []
    recall_list1 = []
    recall_list2 = []
    mrr10_memory_1 = []
    mrr10_memory_2 = []
    recall_memory_1 = []
    recall_memory_2 = []

    print('## Variations Res ##')

    for qv_type in VARIATION_TYPES:
        res1 = '{}/queries.dev.small.{}.q32.p128.top1k.run.txt'.format(setting_1, qv_type)
        res2 = '{}/queries.dev.small.{}.q32.p128.top1k.run.txt'.format(setting_2, qv_type)
        mrr_ten1, mrr_dict1, recall_1, recall_dict_1, q_num1 = calculate_mrr_and_recall(qrels, res1)
        mrr_ten2, mrr_dict2, recall_2, recall_dict_2, q_num2 = calculate_mrr_and_recall(qrels, res2)
        assert q_num1 == q_num2
        mrr10_memory_1.append(mrr_ten1)
        mrr10_memory_2.append(mrr_ten2)
        recall_memory_1.append(recall_1)
        recall_memory_2.append(recall_2)

        print('# {} #'.format(qv_type))
        print('[mrr_10: {0}] [recall: {1}] [Q_num: {2}]'.format(str(mrr_ten1), str(recall_1), str(q_num1)))
        print('[mrr_10: {0}] [recall: {1}] [Q_num: {2}]'.format(str(mrr_ten2), str(recall_2), str(q_num2)))
        mrr10_memory_list1 = []
        mrr10_memory_list2 = []
        for qid in mrr_dict1.keys():
            mrr10_memory_list1.append(mrr_dict1[qid]['mrr_10'])
            mrr10_memory_list2.append(mrr_dict2[qid]['mrr_10'])
        mrr10_memory_p_value = stats.ttest_rel(mrr10_memory_list1, mrr10_memory_list2)[1]

        recall_memory_list1 = []
        recall_memory_list2 = []
        for qid in recall_dict_1.keys():
            recall_memory_list1.append(recall_dict_1[qid]['recall'])
            recall_memory_list2.append(recall_dict_2[qid]['recall'])
        recall_memory_p_value = stats.ttest_rel(recall_memory_list1, recall_memory_list2)[1]
        print('tt_test_p_value: [mrr_10: {}] [recall: {}]'.format(str(mrr10_memory_p_value), str(recall_memory_p_value)))

        for qid in mrr_dict1.keys():
            mrr10_list1.append(mrr_dict1[qid]['mrr_10'])
            mrr10_list2.append(mrr_dict2[qid]['mrr_10'])

        for qid in recall_dict_1.keys():
            recall_list1.append(recall_dict_1[qid]['recall'])
            recall_list2.append(recall_dict_2[qid]['recall'])

    avg_mrr10_1 = sum(mrr10_list1) / len(mrr10_list1)
    avg_mrr10_2 = sum(mrr10_list2) / len(mrr10_list2)
    avg_recall_1 = sum(recall_list1) / len(recall_list1)
    avg_recall_2 = sum(recall_list2) / len(recall_list2)

    assert round(sum(mrr10_memory_1) / len(mrr10_memory_1), 4) == round(avg_mrr10_1, 4)
    assert round(sum(mrr10_memory_2) / len(mrr10_memory_2), 4) == round(avg_mrr10_2, 4)
    assert round(sum(recall_memory_1) / len(recall_memory_1), 4) == round(avg_recall_1, 4)
    assert round(sum(recall_memory_2) / len(recall_memory_2), 4) == round(avg_recall_2, 4)

    print('# Variation Avg #')
    print('Avg: [mrr_10: {}] [recall: {}]'.format(str(avg_mrr10_1), str(avg_recall_1)))
    print('Avg: [mrr_10: {}] [recall: {}]'.format(str(avg_mrr10_2), str(avg_recall_2)))

    mrr10_p_value = stats.ttest_rel(mrr10_list1, mrr10_list2)[1]
    recall_p_value = stats.ttest_rel(recall_list1, recall_list2)[1]
    print('tt_test_p_value: [mrr_10: {}] [recall: {}]'.format(str(mrr10_p_value), str(recall_p_value)))

    mrr10_p_value_memory = stats.ttest_rel(mrr10_memory_1, mrr10_memory_2)[1]
    recall_p_value_memory = stats.ttest_rel(recall_memory_1, recall_memory_2)[1]
    print('type_tt_test_p_value: [mrr_10: {}] [recall: {}]'.format(str(mrr10_p_value_memory), str(recall_p_value_memory)))
    print('#################################################################')


if __name__ == '__main__':
    argv = sys.argv
    if argv[4] == 'fusion':
        fusion_tt_test(argv[1], argv[2], argv[3])
    else:
        tt_test(argv[1], argv[2], argv[3])