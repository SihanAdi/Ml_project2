from matplotlib import pyplot as plt


def scatter_plot(train_data, feature):
    plt.scatter(train_data[feature], train_data['price'], s=15, color="red")
    plt.title(feature)
    plt.show()


def scatter_plot_SVM(data, y):
    list = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]

    for i in range(len(list)):
        j = i + 1
        while j < len(list):
            plt.scatter(data[list[i]], data[list[j]], c=y, cmap=plt.cm.Paired)
            plt.title("Visuals for data")
            plt.xlabel(list[i])
            plt.ylabel(list[j])
            plt.show()
            j+=1


def correlation_bar_chart(corr):
    corr.sort_values().plot.bar()
    plt.show()


def figure(title, *datalist):
    plt.figure()
    for v in datalist:
        plt.plot(v[0], '-', label=v[1], linewidth=2)
        plt.plot(v[0], 'o')
    plt.grid()
    plt.title(title)
    plt.show()


