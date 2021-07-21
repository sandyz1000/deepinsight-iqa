
# Flag to test in bulk or for single file

BULK_TESTING = False

# Source path to read images

READFROM = '/home/ssg0283/Documents/ImageData/01-10-data/IQA/test'

# Image Path If Bulk testing flag False

IMAGEPATH = '/home/ssg0283/Documents/ImageData/01-10-data/IQA/GOOD/4016_3000CD0118873_CANCELLED_CNCLD_PAN.jpg'


# path to save TP/FN/TN/FP images in folder

OUTPUTPATH = '/home/ssg0283/Documents/ImageData/01-10-data/IQA/conftest'

"""
====================================================================================
"""

# Threshold values

conf_matrix_thresholds = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]

conf_matrix_thresholds_v2 = [34.1, 34.2, 34.3, 34.4, 34.5, 34.6, 34.7, 34.8, 34.9]

conf_matrix_thresholds_v3 = [35.1, 35.2, 35.3, 35.4, 35.5, 35.6, 35.7, 35.8, 35.9]

"""
====================================================================================
"""


def get_cnf_Matrix_label(prd_score, exp_score, image_label):
    prd_score = round(prd_score)
    label = ''
    if image_label == 'BAD':
        if prd_score <= exp_score:
            label = 'TN'
        else:
            label = 'FP'
    if image_label == "GOOD":
        if prd_score >= exp_score:
            label = 'TP'
        else:
            label = 'FN'
    return prd_score, label


def get_response(stub, img, result_dict, file, score, image_lable):
    img_h, img_w, img_ch = img.shape
    iqa_resp = stub.VerifyImageFitnessBrisque(iqa_pb2.iqa_request(
        image=img.flatten().tostring(),
        img_width=img_w,
        img_height=img_h,
        img_channels=img_ch
    ))

    #print(f"Image IQA passed? : {iqa_resp.fitness} and score is {iqa_resp.score}")
    temp_dict = {'accept': iqa_resp.fitness, 'score': iqa_resp.score}
    result_dict[file] = temp_dict
    prd_score, cnf_mat_label = get_cnf_Matrix_label(iqa_resp.score, score, image_lable)
    return prd_score, cnf_mat_label


def run(img):
    if img is None:
        return img
    channel = grpc.insecure_channel(f"localhost:{SERVER_PORT}")
    stub = iqa_pb2_grpc.IQAServiceStub(channel)
    if BULK_TESTING:

        result_dict = {}
        prifix = 'BAD'
        for score in conf_matrix_thresholds:
            tp_count = 0
            tn_count = 0
            fp_count = 0
            fn_count = 0
            for root, dirs, files in os.walk(READFROM):
                for file in files:
                    try:
                        img = cv2.imread(os.path.join(root, file), cv2.IMREAD_COLOR)
                        image_lable = file.split('.')[0].split('_')[-1]
                        prd_score, cnf_matrix_label = get_response(stub, img, result_dict, file, score, image_lable)
                        if cnf_matrix_label == 'TP':
                            tp_count = tp_count + 1
                        if cnf_matrix_label == 'TN':
                            tn_count = tn_count + 1
                        if cnf_matrix_label == 'FP':
                            fp_count = fp_count + 1
                        if cnf_matrix_label == 'FN':
                            fn_count = fn_count + 1

                        prd_score = str(prd_score).split('.')
                        org_label = str(score)
                        pre_label = str(prd_score[0])
                        path_to_dire = os.path.join(OUTPUTPATH, org_label)
                        path_to_dire = os.path.join(path_to_dire, cnf_matrix_label)
                        if not os.path.exists(path_to_dire):
                            os.makedirs(path_to_dire)
                        image_path = os.path.join(path_to_dire, file + "_" + org_label + "_" + pre_label)
                        cv2.imwrite(image_path, img)
                    except grpc.RpcError as err:
                        status_code = err.code()
                        message = err.details()
                        print(status_code)
                        print(message)

            print("====================================================")
            print("confMatxrix for ", score)
            print("====================================================")

            print('TP : ', tp_count)
            print('FN : ', fn_count)
            print('TN : ', tn_count)
            print('FP : ', fp_count)

            accuracy = (tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count)
            error_rate = (fp_count + fn_count) / (tp_count + tn_count + fp_count + fn_count)
            recall = tp_count / (tp_count + fn_count)
            precision = tp_count / (tp_count + fp_count)
            F1_score = 2 * (recall * precision) / (recall + precision)
            # specificity=tn_count/(tn_count+fp_count)
            print("--------------------------")
            print("Accuracy IQA : ", accuracy)
            print("ErrorRate IQA : ", error_rate)
            print("Sensitivity/Recall/TP_Rate : ", recall)
            print("Precision : ", precision)
            print("F1-Score : ", F1_score)
            print("--------------------------")

    else:
        #filepath = '/home/ssg0283/Documents/ImageData/01-10-data/IQA/GOOD/4016_3000CD0118873_CANCELLED_CNCLD_PAN.jpg'
        img = cv2.imread(IMAGEPATH, cv2.IMREAD_COLOR)
        result_dict = {}
        # result_dict = get_response(stub, img, result_dict, filepath.split('/')[-1], card_type)
        score, buketname = get_response(stub, img, result_dict, filepath.split('/')[-1])
        print(score)
        print(buketname)


if __name__ == "__main__":
    import cv2

    im = cv2.imread("/home/ssg0283/Desktop/IMG_20200106_150125.jpg", cv2.IMREAD_UNCHANGED)
    run(im)
