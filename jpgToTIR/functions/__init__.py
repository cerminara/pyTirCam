import numpy as np
from PIL import Image
import time


def closest_color(rgb, colors):
    """
    Given the rgb tuple and the colors tuple array, this function
    finds the color in colors closest to rgb.

    Input:

    rgb          - (r, g, b) tuple
    colors       - array of rgb tuples in the colorbar

    Output:

    the index of the color in the colorbar closest to rgb

    """
    r, g, b = rgb
    nColors = len(colors[:, 0])
    color_diffs = np.ones(nColors)*1e10
    for i in range(nColors):
        cr, cg, cb = colors[i, :]
        color_diffs[i] = (r - cr)**2 + (g - cg)**2 + (b - cb)**2
    return np.argmin(color_diffs)

def analyticBar(xT):
    """
    Defines the rgb values of a specific color bar.

    Input:

    xT           - array of the colorbar temperature values

    Output:

    xT           - array of the colorbar temperature values
    rgb          - color tuple corresponding to each temperature in xT

    """
    T1 = -10.
    T2 = 0.
    T3 = 10.
    T4 = 18.
    T5 = 28.
    T6 = 38.
    T7 = 48.
    T8 = 60.
    conds = [xT < T2,
             (xT >= T2) & (xT < T3),
             (xT >= T3) & (xT < T4),
             (xT >= T4) & (xT < T5),
             (xT >= T5) & (xT < T6),
             (xT >= T6) & (xT < T7),
             xT >= T7]
    funcr = [lambda xT: 255./(T2-T1)*xT + 255.,
             lambda xT: -255./(T3-T2)*xT + 255.,
             lambda xT: 0.,
             lambda xT: 0.,
             lambda xT: 255./(T6-T5)*(xT-T5),
             lambda xT: -55./(T7-T6)*(xT-T6) + 255.,
             lambda xT: 55./(T8-T7)*(xT-T7) + 200.]
    funcg = [lambda xT: 0.,
             lambda xT: 5.0*xT,
             lambda xT: 195./(T4-T3)*(xT-T3) + 50.,
             lambda xT: -140/(T5-T4)*(xT-T4) + 245.,
             lambda xT: 150./(T6-T5)*(xT-T5) + 105.,
             lambda xT: -255./(T7-T6)*(xT-T6) + 255.,
             lambda xT: 240./(T8-T7)*(xT-T7)]
    funcb = [lambda xT: 250./(T2-T1)*(xT-T1),
             lambda xT: -55./(T3-T2)*(xT-T2) + 250.,
             lambda xT: 45./(T4-T3)*(xT-T3) + 195.,
             lambda xT: -240./(T5-T4)*(xT-T4) + 240.,
             lambda xT: 0.,
             lambda xT: 0.,
             lambda xT: 240./(T8-T7)*(xT-T7)]
    r = np.piecewise(xT, conds, funcr)
    g = np.piecewise(xT, conds, funcg)
    b = np.piecewise(xT, conds, funcb)
    return [xT, np.c_[r, g, b]]


def fromJpgToBar(FileName, TBound, barBound):
    """
    Extract the rgb values of the image colorbar.

    Input:

    FileName     - name of the image file
    TBound       - [minimum, maximum] colorbar temperature
    barBound     - bounds pixel coordinates [bar_wi, bar_wf, bar_hi, bar_hf]

    Output:

    temp         - array of the colorbar temperature values
    avg          - color tuple corresponding to each temperature in xT

    """

    im_tot = Image.open(FileName)
    imarr_tot = np.array(im_tot)

    # Select the colorbar and image zone
    bar_wi=barBound[0]
    bar_wf=barBound[1]
    bar_hi=barBound[2]
    bar_hf=barBound[3]
    n_h = bar_hf - bar_hi
    imarr_bar = imarr_tot[bar_hi:bar_hf, bar_wi:bar_wf, :]

    # Insert temperature range of the image colorbar 
    temp = np.linspace(TBound[0], TBound[1], n_h)

    avg_aus = np.average(imarr_bar, axis=1)
    avg = avg_aus[::-1]
    return [temp, avg]


def fromJpgToArray(FileName, imBound, temp, COLORS):
    """
    Extract the temperature values from the compressed image.

    Input:

    FileName     - name of the image file
    imBound      - bounds pixel coordinates [wi, wf, hi, hf]
    temp         - array of the colorbar temperature values
    COLORS       - color tuple corresponding to each temperature in temp

    Output:

    Tarr         - recovered temperature matrix
    imarr_out    - rgb matrix corresponding to Tarr

    The function prints into the standard output the time needed to convert.

    """

    start_time = time.time()

    im_tot = Image.open(FileName)
    imarr_tot = np.array(im_tot)
    w, h = im_tot.size
    # control that imBound is an array woth size 4
    # imBound = [min_x, max_x, min_y, max_y]

    # Select dimension of the zone of the image
    wi = imBound[0]
    wf = imBound[1]
    hi = imBound[2]
    hf = imBound[3]
    imarr_cut = imarr_tot[hi:hf, wi:wf, :] # selected zone 
    w_im = wf-wi
    h_im = hf-hi

    Tarr = np.zeros((h_im,w_im))
    imarr_out = imarr_cut.copy()
    ### gain obtained using duplicated pixel colors
    #unq_rows, count = np.unique(Simarr.reshape(h_Sim*w_Sim, 3), axis=0, return_counts=1)
    #print("Gain = ", h_Sim*w_Sim/len(count))
    for ii in range(h_im):
        for jj in range(w_im):
            pixel_ij = imarr_cut[ii,jj,:]
            cc_ij = closest_color(pixel_ij, COLORS)
            Tarr[ii,jj] = temp[cc_ij]  
            imarr_out[ii,jj,:] = COLORS[cc_ij, :]
    print("--- Time needed to convert: %s seconds ---" % (time.time() - start_time))
    return [Tarr, imarr_out]
