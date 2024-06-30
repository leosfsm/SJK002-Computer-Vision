#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import glob
import os
import visualPercepUtils as vpu


def histeq(im, nbins=256):
    imhist, bins = np.histogram(im.flatten(), list(range(nbins)), density=False)
    cdf = imhist.cumsum() # cumulative distribution function (CDF) = cummulative histogram
    factor = 255 / cdf[-1]  # cdf[-1] = last element of the cummulative sum = total number of pixels)
    im2 = np.interp(im.flatten(), bins[:-1], factor*cdf)
    return im2.reshape(im.shape), cdf

def darkenImg(im,p=2):
    return ((im ** float(p)) / (255 ** (p - 1))).astype('uint8') # try without the float conversion and see what happens

def brightenImg(im,p=2):
    return np.power(255.0 ** (p - 1) * im, 1. / p).astype('uint8')  # notice this NumPy function is different to the scalar math.pow(a,b)

def checkBoardImg(im, m, n):
    shape = im.shape
    im2 = im.copy()
    horizontal = np.block([np.tile(np.repeat(np.array([-1,1]), n),shape[0]//(2*n)), np.repeat(np.array([-1,1]), n)[:shape[0]%(2*n)]])
    vertical = np.block([np.tile(np.repeat(np.array([1,-1]), m),shape[1]//(2* m)), np.repeat(np.array([1,-1]), m)[:shape[1]%(2*m)]])
    mask = vertical[:,np.newaxis] * horizontal
    im2[mask<0] = 255 - im2[mask<0]
    return im2

def multiHist(im, n, nbins=256):
     hists = []
     for i in range(n):
          quad = [M for SubA in np.array_split(im, i + 1, axis=0) for M in np.array_split(SubA, i + 1, axis=1)]
          hists += [np.histogram(subIm.flatten(), list(range(nbins)), density=False)[0] for subIm in quad]
     return hists

def testDarkenImg(im):
    im2 = darkenImg(im,p=2) #  Is "p=2" different here than in the function definition? Can we remove "p=" here?
    return [im2]

def testBrightenImg(im):
    p=2
    im2=brightenImg(im,p)
    return [im2]

def testHistEq(im):
    im2, cdf = histeq(im)
    return [im2, cdf]

def testCheckBoardImg(im):
    im2 = checkBoardImg(im, 64, 64)
    return [im2]

def testMultiHist(im):
    hists = multiHist(im, 2)
    return [hists]

def saveImg(image, path):
    pil_im = Image.fromarray(image.astype(np.uint8))  # from array to Image
    pil_im.save(path)

path_input = './imgs-P1/'
path_output = './imgs-out-P1/'
bAllFiles = True
bAllTests = False
bSaveResultImgs = True
nameTests = {'testHistEq': "Histogram equalization",
             'testBrightenImg': 'Brighten image',
             'testDarkenImg': 'Darken image',
             'testCheckBoardImg': 'Checkboard Image',
             'testMultiHist': 'Multi-Histogramm'}
suffixFiles = {'testHistEq': '_heq',
               'testBrightenImg': '_br',
               'testDarkenImg': '_dk',
               'testCheckBoardImg': '_ckb',
               'testMultiHist': '_mh'}

if bAllFiles:
    files = glob.glob(path_input + "*.pgm")
else:
    files = [path_input + 'iglesia.pgm'] # iglesia,huesos

if bAllTests:
    tests = ['testHistEq', 'testBrightenImg', 'testDarkenImg', 'testCheckBoardImg', 'testMultiHist']
else:
    tests = ['testMultiHist']


def doTests():
    print("Testing on", files)
    for imfile in files:
        im = np.array(Image.open(imfile).convert('RGB'))  # from Image to array
        for test in tests:
            out = eval(test)(im)
            im2 = out[0]
            if test == 'testMultiHist':
                vpu.showInGrid(im2)
            else:
                vpu.showImgsPlusHists(im, im2, title=nameTests[test])
                if len(out) > 1:
                    vpu.showPlusInfo(out[1],"cumulative histogram" if test=="testHistEq" else None)
                if bSaveResultImgs:
                    dirname, basename = os.path.dirname(imfile), os.path.basename(imfile)
                    fname, fext = os.path.splitext(basename)
                    #print(dname,basename)
                    saveImg(im2, path_output + '//' + fname + suffixFiles[test] + fext)

if __name__== "__main__":
    doTests()

