##############################################################################
##                     Written by Damilola Payne                            ##
##                     Sample of gradient descent                           ##
##############################################################################
# A quick example of gradient descent in python
#f(x)=x**4−3x**3+2, with derivative f'(x)=4x**3−9x**2
#find a local minimum
# From calculation, it is expected that the local minimum occurs at x=9/4

# so how does it work we start with the guess which is our current x value (cur_x)
# we find the gradient of our given point and then aim to minimise the gradient 
# by moving slowly in the direction that increases the gradient by the value gamma
# the result is that we slowly move towards the local minimum


cur_x = 6  # The algorithm starts at x=6
gamma = 0.01  # step size multiplier
precision = 0.00001
previous_step_size = cur_x


def df(x):
    return 4 * x**3 - 9 * x**2


while previous_step_size > precision:
    prev_x = cur_x
    cur_x += -gamma * df(prev_x)
    previous_step_size = abs(cur_x - prev_x)

print("The local minimum occurs at %f" % cur_x)
