# Chú thích a = weight, b = bias

def predict(X,a,b):
    return a * X + b

def cost_function(X,Y,a,b):
    n = len(X)

    sum_error = 0
    for i in range(n):
        sum_error += (Y[i] - (a * X[i] + b))**2
    
    return sum_error / n

def update_weight(X,Y,a,b,learning_rate):
    n = len(X)

    a_temp = 0
    b_temp = 0

    for i in range(n):
        a_temp += -2 * X[i] * (Y[i] - (a * X[i] + b))
        b_temp += -2 * (Y[i] - (a * X[i] + b))

    a -= (a_temp / n) * learning_rate
    b -= (b_temp / n) * learning_rate

    return a, b

def tran(X,Y,a,b,learning_rate,iter):
    for i in range(iter):
        a, b = update_weight(X,Y,a,b,learning_rate)
        cost = cost_function(X,Y,a,b)
    
    return a,b

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

a,b = tran(x,y,0.03,0.001, 0.001,30)
c = predict(10,a,b)

print(c)