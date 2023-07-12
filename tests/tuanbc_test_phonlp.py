# -*- coding: utf-8 -*-
import phonlp
import py_vncorenlp

def test_phonlp():

    #text = "bầu_trời hôm_nay thật trong_xanh , bát_ngát và nhiều chim"
    text = "Ngành Khoa học Máy tính thuộc nhóm ngành Máy tính và Công nghệ thông tin. Mục tiêu của chương trình ngành Khoa học Máy tính là đào tạo ra những kỹ sư có chất lượng cao, có khả năng thiết kế, xây dựng và triển khai những hệ thống phần mềm đáp ứng nhu cầu trong nước và quốc tế. Kỹ sư tốt nghiệp ngành Khoa học Máy tính cũng được trang bị những kiến thức cần thiết để có thể học tiếp cao học và tiến sỹ trong lĩnh vực Máy tính và Công nghệ thông tin."
    

    # Automatically download VnCoreNLP components from the original repository
    # and save them in some local machine folder
    #py_vncorenlp.download_model(save_dir='E:/git-workspace/PhoNLP/vncorenlp') @#==# ko chạy đc, fai download thủ công

    # Load VnCoreNLP for word and sentence segmentation
    #rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='./vncorenlp')
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='E:/git-workspace/PhoNLP/vncorenlp')

    # Perform word and sentence segmentation
    segmentList = rdrsegmenter.word_segment(text)
    #print(segmentList)
    # 
    segmentText = ' '.join(map(str, segmentList))
    print(segmentText)

    #phonlp.download('./')
    model = phonlp.load("E:/git-workspace/PhoNLP/pretrained_phonlp")
    
    
    # Annotate a word-segmented sentence
    #out = model.annotate(segmentText)
    #model.print_out(out)
    
    # Annotate a corpus where each line represents a word-segmented sentence
    # model.annotate(input_file='./tests/testcase_01.txt',
    #                output_file='./tests/testcase_01_output.txt')
    tmpInputFile = 'E:/git-workspace/PhoNLP/tests/data/testcase_01_mini.txt'
    tmpOutputFile = 'E:/git-workspace/PhoNLP/tests/data/testcase_01_mini_output.txt'
    
    model.annotate(input_file=tmpInputFile,
                   output_file=tmpOutputFile)


if __name__ == '__main__':
    test_phonlp()