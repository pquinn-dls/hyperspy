import numpy as np

def snip2d(image,
                filter_window=13,
                iterations=16):
    """
    use snip filter algorithm to obtain background image

    Parameters
    ----------
    image : array
        intensity image 
    filter_window : int, optional
        window size in indices. Typically twice the fwhm of features in image 
    iterations : int, optional
        Number of iterations in snip filter


    Returns
    -------
    background_image : array

    References
    ----------

    .. [1] C.G. Ryan etc, "SNIP, a statistics-sensitive background
           treatment for the quantitative analysis of PIXE spectra in
           geoscience applications", Nuclear Instruments and Methods in
           Physics Research Section B, vol. 34, 1998.
           
       [2] M Morhac etc, "Background elimination methods for 
           multidimensional coincidence gamma-ray spectra", 
           Nuclear Instruments and Methods 
           in Physics Research Section A, vol. 401, 1997. 113-132
           
    """
    
    def filter_image(image,window_p,row,cols,row_index,col_index):
        # snip 
        # reduce the filter width at the edges as this will cause 
        # large drop-offs otherwise
        row_lo_index = np.where(row_index>=window_p,row_index-window_p,0)
        row_hi_index = np.where(row_index>=window_p,row_index+window_p,
                                row_index+row_index)
        row_hi_index = np.where(row_hi_index<rows,row_hi_index,rows-1)
        row_lo_index = np.where(row_lo_index<rows-2*window_p,row_lo_index,
                                row_index - (rows - row_index) +1)
        col_lo_index = np.where(col_index>=window_p,col_index-window_p,0)
        col_hi_index = np.where(col_index>=window_p,col_index+window_p,
                                col_index+col_index)
        col_hi_index = np.where(col_hi_index<cols,col_hi_index,cols-1)
        col_lo_index = np.where(col_lo_index<cols-2*window_p,col_lo_index,
                                col_index - (cols - col_index) +1)
        
        # +p, +p
        P1 = image[np.ix_(row_hi_index,col_hi_index)]
        # -p, +p
        P2 = image[np.ix_(row_lo_index,col_hi_index)]
        # +p,-p
        P3 = image[np.ix_(row_hi_index,col_lo_index)]
        # -p,-p
        P4 = image[np.ix_(row_hi_index,col_lo_index)]
        #  +p,0
        S1 = image[np.ix_(row_hi_index,col_index)]
        #  0,+p
        S2 = image[np.ix_(row_index,col_hi_index)]
        #  0,-p
        S3 = image[np.ix_(row_index,col_lo_index)]
        #  0,-p
        S4 = image[np.ix_(row_lo_index,col_index)]
        
        S1 = np.maximum(S1,0.5*(P1+P3))
        S2 = np.maximum(S2,0.5*(P1+P2))
        S3 = np.maximum(S3,0.5*(P3+P4))
        S4 = np.maximum(S4,0.5*(P2+P4))
        
        S1 = S1 - 0.5*(P1+P3)
        S2 = S2 - 0.5*(P1+P2)
        S3 = S3 - 0.5*(P3+P4)
        S4 = S4 - 0.5*(P2+P4)
        
        a = (S1 +S4)/2.0 + (S2+S3)/2.0 +  (P1+P2+P3+P4)/4.0
    
        new_image = np.minimum(image,a)

        return new_image
    
    smooth_image = np.log(np.log(image + 1) + 1)
    rows,cols = smooth_image.shape
    window_p = int(filter_window)
    row_index      = np.arange(rows)
    col_index      = np.arange(cols)

    for p in range(iterations):
        smooth_image = \
            filter_image(smooth_image,window_p,rows,cols,row_index,col_index)
         
    # a final smoothing/filter with a reducing step size.
    for i in range(window_p+10):        
        smooth_image = \
            filter_image(smooth_image,window_p,rows,cols,row_index,col_index)
        window_p= max(window_p-1,1)
        
    background_image = np.exp(np.exp(smooth_image) - 1) - 1
    inf_ind = np.where(~np.isfinite(background_image))    
    background_image[inf_ind] = 0.0
    
    return image-background_image
    

