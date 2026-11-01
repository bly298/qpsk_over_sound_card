import matplotlib.pyplot as plt


def refined_constellation_show(symbols, num_to_plot, title=''):
    x = [0.0] * num_to_plot
    y = [0.0] * num_to_plot

    for i in range(num_to_plot):
        x[i] = symbols[i].real
        y[i] = symbols[i].imag

    plt.plot(x, y, color='blue', linestyle='', marker='.')
    plt.title(title)
    plt.grid()
    plt.show()
    return