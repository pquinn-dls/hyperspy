
import numpy as np
from scipy import optimize
from copy import copy
#import pybobyqa


def Similarity_ErrFunc(param,ref_images,moving_images,interpolation_order,fill_value,bin_rule='auto',measure="MI"):
    
    if isinstance(ref_images, np.ndarray):
        ref_images    = [ref_images]
        moving_images = [moving_images]
    
    mi_sum=0.0
    for ref,move in zip(ref_images,moving_images):
        # Perform a translational shift
        _img = rigid_shift_image(move,param,interpolation_order, fill_value)
        # crop both images....
        n_y, n_x = _img.shape
        y, x = -int(np.round(param[0])), -int(np.round(param[1]))
        top, bot = max(0, y), min(n_y, n_y+y)
        lef, rig = max(0, x), min(n_x, n_x+x),
        _img_crop = _img[ top-y:bot-y, lef-x:rig-x]
        ref_crop  = ref[ top-y:bot-y, lef-x:rig-x]
        # compare the images
        mi_sum+=similarity_measure(ref_crop,_img_crop,bin_rule=bin_rule,measure=measure)
    # measure the mutual information between the reference and moving image
    return 1.0/mi_sum


#def estimate_image_shift_MI(img,ximg,interpolation_order,fill_value,bin_rule='auto',measure="MI"):
#    # Rough search...
#    # rough estimate of number of bins based on the starting image size...
#    #print("img0",img[0].shape)
#    xsize,ysize = img.shape
#    xspan = int(0.3*xsize)
#    yspan = int(0.3*ysize)
#    
#
#    x_range = np.array(range(-xspan,xspan),np.float64)
#    y_range = np.array(range(-yspan,yspan),np.float64)
#    bestx = 0.
#    besty = 0.
#    besterr = 1.e10
#    for xpos in x_range:   
#        for ypos in y_range:
#            err = Similarity_ErrFunc([xpos,ypos],img,ximg,interpolation_order,\
#                                     fill_value,bin_rule='auto',measure="MI")
#            if err < besterr:
#                besterr=err
#                bestx = xpos
#                besty = ypos
#    param = [bestx,besty]
#    param = np.array(param,np.float64)
#    lower = np.array([-xspan, -yspan])
#    upper = np.array([xspan, yspan])
#
#    # Call Py-BOBYQA (with bounds)
#    soln = pybobyqa.solve(Similarity_ErrFunc,param,\
#                          args=(img,ximg,interpolation_order,fill_value,bin_rule),
#                          bounds=(lower,upper))
#    #param = optimize.fmin_l_bfgs_b(MIErrFunc,param,approx_grad=True,epsilon=0.05,
#    #                               args=(img,ximg,interpolation_order,fill_value,bins), bounds=[(-14,14),(-14,14)])        
#    return soln.x,soln.f
#
#
#        

