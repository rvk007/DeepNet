import matplotlib.pyplot as plt 

def plot_metric(values, metric):
    """
    Plot the graph of the given metric

    Arguments:
        values : List of data to be plot
        metric : The metric of data [Loss/Accuracy]
    """

    # Initialize a figure
    fig = plt.figure(figsize=(7, 5))

    # Plot values
    plt.plot(values)

    # Set plot title
    plt.title(f'Validation {metric}')

    # Label axes
    plt.xlabel('Epoch')
    plt.ylabel(metric)

    # Set legend
    location = 'upper' if metric == 'Loss' else 'lower'

    # Save plot
    fig.savefig(f'{metric.lower()}_change.png')