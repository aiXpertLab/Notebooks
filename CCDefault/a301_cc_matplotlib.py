import matplotlib.pyplot as plt


def plotting(first, second, firstlabel='Training loss', secondlabel='Validation loss'):

    fig = plt.figure(figsize=(15, 5))
    plt.plot(first, label=firstlabel)
    plt.plot(second, label=secondlabel)
    plt.legend(frameon=False, fontsize=15)
    plt.show()
