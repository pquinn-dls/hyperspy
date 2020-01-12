
import numpy as np
from scipy import optimize
#import pybobyqa


def MI_ErrFunc(param,ref_images,moving_images,interpolation_order,fill_value,bin_rule='auto',measure="MI"):
    
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



def translate_register(static_image,moving_image,errFunc,runCoarse=True,opt='global',
                      rotation=False):
    # Rough search...
    # rough estimate of number of bins based on the starting image size...
    xsize,ysize = img[0].shape
    # 
  	xspan = int(0.3*xsize)
    yspan = int(0.3*ysize)

    if opt=='global':
        bounds=[(-xspan,xspan),(-yspan,yspan)]
        param = optimize.differential_evolution(errFunc,bounds,args=(static_image,moving_image,bins))
        param = param.x
        _img = rigid_shift_image(param,moving_image)
    else:
        x_range = np.array(range(-xspan,xspan))
        y_range = np.array(range(-yspan,yspan))
        bestx = 0
        besty = 0
        besterr = 1e10
        for xpos in x_range:
            for ypos in y_range:
                err = errFunc([xpos,ypos],static_image,moving_image)
                if err < besterr:
                    besterr=err
                    bestx = xpos
                    besty = ypos
        param = [bestx,besty]
        param = optimize.fmin(errFunc,param,args=(static_image,moving_image,bins))
        
    return (_img,param)
#
#
#def estimate_image_shift_MI(img,ximg,interpolation_order,fill_value,bin_rule='auto',measure="MI"):
#    # Rough search...
#    # rough estimate of number of bins based on the starting image size...
#    #print("img0",img[0].shape)
#    xsize,ysize = img.shape
#  	xspan = int(0.3*xsize)
#    yspan = int(0.3*ysize)
#    
#
#    x_range = np.array(range(-xpan,xspan),np.float64)
#    y_range = np.array(range(-yspan,yspan),np.float64)
#    bestx = 0.
#    besty = 0.
#    besterr = 1.e10
#    for xpos in x_range:   
#        for ypos in y_range:
#            err = MI_ErrFunc([xpos,ypos],img,ximg,interpolation_order,fill_value,bin_rule='auto',measure="MI")
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
#    soln = pybobyqa.solve(MIErrFunc,param,args=(img,ximg,interpolation_order,fill_value,bins),
#                          bounds=(lower,upper))
#    #param = optimize.fmin_l_bfgs_b(MIErrFunc,param,approx_grad=True,epsilon=0.05,
#    #                               args=(img,ximg,interpolation_order,fill_value,bins), bounds=[(-14,14),(-14,14)])        
#    return soln.x,soln.f


def estimate_stack_shifts_MI(signal_dict,signal_list,interpolation_order=1,
                             fill_value=-1.0,mi_threshold=0.2,reference='last',bin_rule='auto',measure="MI"):
    stack = [signal_dict[sig] for sig in signal_list if isinstance(signal_dict[sig],Signal2D)]
    ref = None
    shifts = []
    mi_list=[]
    show_progressbar= True
    pbar_size = len(stack[0])
    pbar = progressbar(total=pbar_size, leave=True, disable=not show_progressbar)
    
    for stack_item in reversed(list(zip(*stack))):
        if ref is None:
            ref = copy(stack_item)
            ref = [st.data for st in ref]
            shift = np.array([0.0, 0.0])
            shifts.append(shift.copy())
            pbar.update(1)
            continue
        im = [st.data for st in stack_item]
        nshift,mi = estimate_image_shift_MI(ref,im,interpolation_order,fill_value,bin_rule='auto',measure="MI")
        mi_list.append(mi)
        if reference == 'cascade':
            shift += nshift
            ref = copy(im)
        else:
            #if(mi > 1./mi_threshold):
            #    shift = copy(shifts[-1])
            #else:
            shift = nshift
        shifts.append(shift.copy())
        pbar.update(1)

    shifts = np.array(shifts)
    mi_arr = np.array(mi_list)
    shifts = -shifts[::-1]
    mi_arr = mi_arr[::-1]
    del ref
    return shifts,mi_arr        
        

def smart_align_images(ref_image_stack,moving_image,direction_axis=0):

    # align _rigid

    # image gradients,
    ref_xg,ref_yg=np.gradient(ref_image)
    move_xg,move_yg = np.gradient(moving_image)

    ref_xg = ref_xg/np.linalg.norm(ref_xg)
    ref_yg = ref_yg/np.linalg.norm(ref_yg)

    move_xg = move_xg/np.linalg.norm(move_xg)
    move_yg = move_yg/np.linalg.norm(move_yg)
    #
    #
    #    
    delta = -np.matmul((moving_image - ref_image),(ref_xg + move_xg).T)

    sum_delta = sum_delta + alpha*delta
    new_ref = np.matmul(sum_delta,ref_image.T)


