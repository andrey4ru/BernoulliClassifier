def FPR(predict, target, threshold, class_numb):
    FP = 0  # false positive
    CN = 0  # condition negative
    for i in range(len(predict)):
        if target[i] != class_numb:
            CN += 1  # calculate number of condition negative
            if predict[i] >= threshold:
                FP += 1  # calculate number of false positive
    return FP/CN  # return false positive rate
