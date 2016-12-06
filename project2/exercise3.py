from numpy import dot


def PRank(training_data):
    # receives a list of tuples. Each tuple is composed of a list of floats and a boolean (list, boolean)
    # the list of floats contains the candidates' features and the boolean is True if the candidate is a keyword

    # returns the weight vector and the threshold

    w = [0] * len(training_data[0][0]) #weight vector
    b = 0 #threshold

    for x in training_data:
        if (dot(w, x[0]) < b):
            yp = False
        else:
            yp = True

        y = x[1]

        if(yp != y):

            if(y==False):
                yr = -1
            else:
                yr = 1

            if(yr*(dot(w,x[0])-b) <= 0):
                for i in range(0, len(w)):
                    w[i] = w[i] + yr * x[0][i]
                b -= yr

    return w, b
