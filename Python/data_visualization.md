# MATPLOTLIB

# DEPENDENCIES

imports the **matplotlib** dependency to create plots
```python
import matplotlib.pyplot as plt
```

# IMPORTING
loads **image** and displays it
```python
img = plt.imread('image_file.png')

# removes the axis on the image
plt.axis('off')

# displays the image
plt.imshow(img)
```

# EXPORTING
**exports** the plot
```python
# saving as a 'png' file
plt.savefig('image_name.png')
```

# SIZING ATTRIBUTES
resizes the **x and y limits** of the graph
```python
# x limit only
plt.xlim(x_min, x_max)

# y limit only
plt.ylim(y_min, y_max)

# x and y limits together
plt.axis((x_min, x_max, y_min, y_max))

# axis('equal') = equal scaling on x,y axes
# axis('square') = forces square plot
# axis('tight') = sets xlim(), ylim() to show all data
```

improves the **spacing** between subplots
```python
plt.tight_layout()
```
# GRAPHING ATTRIBUTES
determines the style of graph to be used to plot
```python
# use 'plt.style.available' to view a list of available styles
plt.style.use('ggplot')
```

creates a **legend** for the plot; placement is determined by the 'loc' parameter
```python
# use the 'label' parameter in each plt command to specify a specific legend label
plt.legend(loc='upper right')
```

creates a **title** for the plot
```python
plt.title('Title of Plot')
```

creates a label for the **x-axis**
```python
plt.xlabel('Label for X-Axis')
```

creates a label for the **y-axis**
```python
plt.ylabel('Label for Y-Axis')
```

# GRAPHING
plots multiple **line graphs**
```python
plt.plot(x, y, color='blue')

# can plot another line on same graph
plt.plot(x, y_2, color='red')

plt.show()
```

**subplot** automatically determines the layout for multiple plots; plots are placed according to parameters 
```python
# 1=nrows, 2=ncols, nsubplot=1
plt.subplot(1, 2, 1)
plt.plot(x, y)

# the second subplot would require nsubplot=2; nsubplot starts at the top left corner
plt.subplot(1, 2, 2)
plt.plot(x, y_1)
```
---

# SEABORN

# IMPORTS
imports the **seaborn** dependency to create plots with pandas DataFrames
```python
# seaborn is built on top of matplotlib; plt.show() displays seaborn plots
import seaborn as sns
```

# GRAPHING
plots a linear regression graph
```python
# data parameter is required, and x/y must be in string format
sns.lmplot(x='Column_Name', y='Column_Name_2', data=df)

# 'order' parameter dictates higher order regressions
sns.lmplot(x='Column_Name', y='Column_Name_2', data=df, order=2)

# 'hue' parameter groups subsets of data on the same plot
sns.lmplot(x='Column_Name', y='Column_Name_2', data=df, hue='Column_Name_3')

# 'col' and 'row' parameters create subplots in the respective layout
sns.lmplot(x='Column_Name', y='Column_Name_2', data=df, col='Column_Name_3')

# .regplot() is less restrictive than .lmplot()
sns.regplot()
```

plots the **residuals of a regression**; residual structure suggests a simple/single linear regression is not appropriate
```python
sns.residplot(x='Column_Name', y='Column_Name_2', data=df, color='red')
```

**strip plot** displays values on a number line to visualize samples of a single variable
```python
# using the 'x' parameter will provide a grouped strip plot
sns.stripplot(y='Column_Name', data=df)

# jitter spreads out split plot points (by size parameter) so they don't overlap
sns.stripplot(y='Column_Name', data=df, size=num, jitter=True)
```

**swarmplot** automatically arranges repeated points to avoid overlap and provide a sense of distribution
```python
sns.swarmplot(x='Column_Name', y='Column_Name_2', data=df)

# 'hue' parameter displays the colors of a categorial column within the plot
sns.swarmplot(x='Column_Name', y='Column_Name_2', data=df, hue='Column_Name_3')
```

**boxplot** displays the minimum, maximum, and median values of a dataset along the 1st and 3rd quartiles and outliers
```python
sns.boxplot(x='Column_Name', y='Column_Name_2', data=df)
```

**violinplots** show curved distributions (KDE) wrapped around a box plot 
```python
# the distribution is denser where the violin plot is thicker
sns.violinplot(x='Column_Name', y='Column_Name_2, data=df)

```

**jointplots** provide a plot with histograms (of individual coordinates) above and to the side of the main plot
```python
sns.jointplot(x='Column_Name', y='Column_Name', data=df)
```

**pairplots** provide jointplots for all possible pairs of numerical column variables in a df
```python
sns.pairplot(df)
```

**heatmaps** plot covariance matrices when pairplot() plots become visually overwhelming
```python
sns.heatmap(cov_matrix)
```

