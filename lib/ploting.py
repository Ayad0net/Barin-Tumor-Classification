import matplotlib.pyplot as plt


def imshow2d(img1, img2):
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img1)
    axarr[1].imshow(img2, alpha=0.5)
    plt.savefig("fig.pdf")


def imshow2d_overlay(img1, img2):
    plt.figure()
    plt.imshow(img1)
    plt.imshow(img2, alpha=0.5)
    plt.savefig("fig2.pdf")
