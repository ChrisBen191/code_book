# Data Visualization <!-- omit in toc -->

---

# Table of Contents <!-- omit in toc -->

- [MATPLOTLIB](#matplotlib)
  - [Commands](#commands)
  - [Graphing Attributes](#graphing-attributes)
  - [Graphing](#graphing)
- [Seaborn](#seaborn)
  - [Graphing Attributes](#graphing-attributes-1)
  - [Graphing](#graphing-1)
  - [Axes Grids](#axes-grids)
- [Bokeh](#bokeh)
  - [Dependencies](#dependencies)
  - [Exporting](#exporting)
  - [Graphing](#graphing-2)
  - [Modifier Commands](#modifier-commands)

## MATPLOTLIB

### Commands

|                           Command |                                                 |
| --------------------------------: | ----------------------------------------------- |
| `import matplotlib.pyplot as plt` | imports matplotlib.                             |
|              `%matplotlib inline` | added to jupyter notebooks to show in notebook. |
|             `plt.style.available` | displays all available graph styles.            |
|         `plt.style.use('ggplot')` | sets the graph style to be used in plotting.    |

loads an image and displays it

```python
img = plt.imread('image_file.png')

# removes axis on the image and displays
plt.axis('off')
plt.imshow(img)

# saving as a 'png' file
plt.savefig('image_file_copy.png')
```

### Graphing Attributes

|                                  Command |                                                                                                                          |
| ---------------------------------------: | ------------------------------------------------------------------------------------------------------------------------ |
|             `plt.title('Title of Plot')` | creates a **title** for the plot                                                                                         |
|         `plt.xlabel('Label for X-Axis')` | creates a label for the **x-axis**                                                                                       |
|         `plt.ylabel('Label for Y-Axis')` | creates a label for the **y-axis**                                                                                       |
|                 `plt.xlim(x_min, x_max)` | resizes the **x limits** of the graph                                                                                    |
|                 `plt.ylim(y_min, y_max)` | resizes the **y limits** of the graph                                                                                    |
| `plt.axis((x_min, x_max, y_min, y_max))` | resizes the **x and y limits** of the graph, can use 'equal', 'square' or 'tight' parameters                             |
|                `plt.margins(buffer_pct)` | creates a **margin** around the plot to keep data off of the edges                                                       |
|                     `plt.tight_layout()` | improves the **spacing** between subplots                                                                                |
|          `plt.legend(loc='upper right')` | creates a **legend** for the plot; use the 'label' parameter in each **plt** command to specify a different legend label |

plots a **vertical/horziontal line** on the plot, typically used to indicate **percentiles**

```python
# plots a VERTICAL line, when using matplotlib axes
ax.axvline(x=df['Column_Name'], color='r', label='Column')

# plots a HORIZONTAL line, when using matplotlib axes
ax.axhline(x=df['Column_Name'], color='r', label='Column')

# can include LINESTYLE and LINEWIDTH PARAMETERS in addition to others

# call legend to display label
ax.legend()

plt.show()
```

### Graphing

creates a **figure** with the number of **axes** specified

```python
# creates a plot with 1row/2columns that share y-axis label
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True)

# plot SEABORN distribution on AX0
sns.distplot(df['Column_Name'], ax=ax0)

# setting axis specific values for AX0
ax0.set(title, xlabel, ylabel, xlim, ylim, ...)

# plot SEABORN distribution on AX1
sns.distplot(df['Column_Name_2'], ax=ax1)

# setting axis specific values for AX1
ax1.set(title, xlabel, ylabel, xlim, ylim, ...)

# add additional axes (ax2, ax3,...) and adjust nrows/ncols to added plots as needed
plt.show()
```

plots multiple **line graphs**

```python
plt.plot(x, y, color='blue')

# can plot another line on same graph
plt.plot(x, y_2, color='red')

plt.show()
```

plots a **histogram** with the bins specified

```python
# can  pass an integer (bins=10) or an interval for bins instead (bins = [1-2, 3-4, ...])
plt.hist(array, bins=bins)
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

creates an **ECDF** (Empirical Cumulative Distribution Function). The **ECDF** allows you to plot a feature of your data, in order from least to greatest, to see the whole featrure as if it's distributed across the data set.

```python
# need to sort the x data to propertly plot the ECDF
x = np.sort(df['Column_Name'])

# creates a y-axis with evenly spaced data points (with a maximum of one)
y = np.arange(1, len(x)+1) / len(x)

# this will plot data points only without a line connecting the data
plt.plot(x, y, marker='.', linestyle='none')

# provides a margin keep the data off of the edges
plt.margins(0.02)

# read from the x-value, then the y-value determines the percentage of the data that is less than the chosen x-value
plt.show()
```

creates a **Binomial CDF** plot providing a set of probabilities of discrete outcomes. Numpy's version of **Bernoulli Trials**

```python
# n= number of Bernoulli trials, p=probability of success, size= # of times to repeat the experiement
samples = np.random.binomial(n, p, size=int)

# using the 'ecdf' custom function
x, y = ecdf(samples)

# plots the Binomial CDF with markers and no lines
plt.plot(x, y, marker='.', linestyle='none')

plt.show()
```

creates a **Piosson CDF**

```python
'''
Poisson Process = the timing of the next event is completely independent of of when the previous event happened
Poisson Distributed = the average number of arrivals of a Poisson process in a given amount of time; limit of the Binomial distribution for low probability of success and large number of trials
'''

# r = mean of the Poisson distribution (r = n*p), size = # of times to repeat the experiement
samples = np.random.poisson(r, size=int)

# using the 'ecdf' custom function
x, y = ecdf(samples)

# plots the Poisson CDF with markers and no lines
plt.plot(x, y, marker='.', linestyle='none')

plt.show()
```

creates a **Normal PDF**

```python
# compute the mean for the normal distribution
mean = np.mean(array)

# compute the std for the normal distrubution
std = np.std(array)

# parametrizes the normal distrubution; size = # of times to repeat the experiement
samples = np.random.normal(mean, std, size=int)

# plots a histogram of the samples with the # of bins specified
plt.hist(samples, normed=True, histtype='step', bins=int)

plt.show()
```

creates a **Normal CDF** and compares it to the **ECDF** of the data

```python
# compute the mean for the normal distribution
mean = np.mean(array)

# compute the std for the normal distrubution
std = np.std(array)

# parametrizes the normal distrubution; size = # of times to repeat the experiement
samples = np.random.normal(mean, std, size=int)

# computes the ECDF of the data
x, y = ecdf(array)

# computes the Normal ECDF
x_theor, y_theor = ecdf(samples)

# plots the Normal ECDF
plt.plot(x_theor, y_theor)

# plots the ECDF
plt.plot(x, y, marker='.', linestyle='none')

plt.show()
```

---

## Seaborn

### Graphing Attributes

|               Command                |                                                                                 |
| :----------------------------------: | ------------------------------------------------------------------------------- |
|       `import seaborn as sns`        | imports seaborn.                                                                |
| `sns.despine(left=True, right=True)` | removes spines of the plot; add parameters of spines as needed                  |
|        `sns.color_palette()`         | returns the current color palette in use                                        |
|           `sns.palplot()`            | displays the color palettes in jupyter notebook                                 |
|       `sns.set_style('dark')`        | sets the default SEABORN theme (white/dark, whitegrid/darkgrid, ticks)          |
|     `sns.set(color_codes=True)`      | sets plots to use matplotlib color codes                                        |
|      `sns.set_palette(palette)`      | sets the SEABORN color palette (deep, muted, pastell, bright, dark, colorblind) |

### Graphing

plots a barplot, which provides the stats of a categorical column specified

```python
# provides a horizontal bar plot with the CATEGORICAL_COLUMN values on the y-axis; switch axes as needed
sns.barplot(x='Column_Name', y='Categorical_Column', data=df)

plt.show()
```

plots a **distplot**, which provides a histogram and kde density curve, as well as distribution with _rug=True_

```python
# creates a histogram with kde=False
sns.distplot( df['Column_Name'], kde=False, bins=int)

# creates a kde curve with hist=False
sns.distplot( df['Column_Name'], hist=False, bins=int)

# creates a rug plot with rug=True to view the distribution of data
sns.distplot( df['Column_Name'], rug=True, bins=int)

# uses the KEYWORDS_DICT to shade the area under the kde curve
sns.distplot( df['Column_Name'], kde_kws={ shade':True})

plt.show()

```

plots a **linear regression** graph

```python
# plots a single linear regression model fit to data specified
sns.regplot(x='Column_Name', y='Column_Name2', data=df)

# change the polynomial regression using the ORDER parameter
sns.regplot(x='Column_Name', y='Column_Name2', data=df, order=2)

plt.show()
```

plots a **residual plot**, useful for evaluating the fit of a model

```python
sns.residplot(x='Column_Name', y='Column_Name2', data=df, color='red')

# change the polynomial regression using the ORDER parameter
sns.residplot(x='Column_Name', y='Column_Name2', data=df, order=2)

plt.show()
```

**strip plot** plots the scatterplot of a categorical variable

```python
# using the 'x' parameter will provide a grouped strip plot
sns.stripplot(y='Column_Name', data=df)

# jitter spreads out split plot points (by size parameter) so they don't overlap
sns.stripplot(y='Column_Name', data=df, size=num, jitter=True)

plt.show()
```

**swarmplot** plots the scatterplot of a categorical variable; prevents points from overlapping

```python
sns.swarmplot(x='Column_Name', y='Column_Name2', data=df)

# 'hue' parameter displays the colors of a categorial column within the plot
sns.swarmplot(x='Column_Name', y='Column_Name2', data=df, hue='Column_Name3')

plt.show()
```

**boxplot** displays the minimum, maximum, and median values of a dataset along the 1st and 3rd quartiles and outliers

```python
sns.boxplot(x='Categorical Column', y='Column_Name2', data=df)

plt.show()
```

**KDE plots** show curved distributions

```python
sns.kdeplot(data_array)

plt.show()
```

**violinplots** show curved distributions (KDE) wrapped around a box plot

```python
# the distribution is denser where the violin plot is thicker
sns.violinplot(x='Column_Name', y='Column_Name2, data=df)

# inner=None will simplify the plot and remove data points

plt.show()
```

**lvplots** show a hybrid between a boxplot and violinplot and is relatively quick to render

```python
# the distribution is denser where the lvplot is thicker
sns.lvplot(x='Column_Name', y='Column_Name2', data=df)

plt.show()
```

**pairplots** provide jointplots for all possible pairs of numerical column variables in a df

```python
sns.pairplot(df)

plt.show()
```

**heatmaps** plot covariance matrices when pairplot() plots become visually overwhelming

```python
sns.heatmap(df.corr())

plt.show()
```

### Axes Grids

creates a **FacetGrid**, useful for plotting conditional relationships

```python

# creates a FacetGrid for ROW='Column Name';  specify the order of the rows using ROW_ORDER
fg = sns.FacetGrid(df,  row='Categorical Column', row_order=['Cat_Value_1', 'Cat_Value_2', 'Cat_Value_3'])
# use col, col_order to create FacetGrid COLUMN-WISE


# Map a BOXPLOT of specified column onto the grid; can map other PLOTS
fg.map(sns.boxplot, 'Column_Name')

# Show the plot
plt.show()
```

creates a **factorplot**, a simplier way to use a **FacetGrid** for categorical data

```python
# Create a FACTOR PLOT (simplier FacetGrid) that contains BOXPLOTS of the column specified
sns.factorplot(data=df, x='Column_Name', kind='box', row='Categorical Column')
# use col, col_order to create factorplot COLUMN-WISE

plt.show()
```

creates a **lmplot**, which is a **liner regression (regplot)** on a FacetGrid

```python
# data parameter is required, and x/y must be in string format
sns.lmplot(x='Column_Name', y='Column_Name2', data=df)

# 'order' parameter dictates higher order regressions
sns.lmplot(x='Column_Name', y='Column_Name2', data=df, order=2)

# 'hue' parameter groups subsets of data on the same plot
sns.lmplot(x='Column_Name', y='Column_Name2', data=df, hue='Categorical_Column')

# 'col' and 'row' parameters create subplots in the respective layout
sns.lmplot(x='Column_Name', y='Column_Name2', data=df, col='Categorical_Column')

plt.show()
```

creates a **PairGrid**, which shows pairwise relationships between data elements

```python
# creates a PAIRGRID comparing the VARS specified
g = sns.PairGrid( df,  vars=['ColumnName_1', 'Column_Name_2'])

# specifies the kind of PLOT in the diagonal / off-diagonal
g2 = g.map_diag(plt.hist)
g3 = g2.map_offdiag(plt.scatter)

plt.show()
```

creates a **pairplot**, a simplier way to use a **PairGrid**

```python
# creates a PAIRPLOT (simplier PairGrid) comparing the VARS specified
sns.pairplot(data=df,
        vars=['ColumnName_1', 'ColumnName_2'],
        kind='scatter',
        hue='Categorical_Column',
        palette='RdBu',					# other palettes are available
        diag_kws={'alpha':.5})		    # other KWs are available

plt.show()
```

creates a **JointGrid**, which combines **univariate** plots (histogram, kde, rug) with **bivariate** plots (scatter, regression)

```python
# create a JOINTGRID comparing the X/Y VALUES specified
g = sns.JointGrid(x='Column_Name', y='Column_Name2', data=df)

# MAIN-PLOT, MARGINAL-PLOT
g.plot(sns.regplot, sns.distplot)

plt.show()
```

creates a **jointplot**, a simplier way to use a **JointGrid**

```python
# creates a JOINTPLOT w/ a 2ND ORDER POLYNOMIAL REGRESSION
sns.jointplot(x='X_axis_column',
         y='Y_axis_column',
         kind='reg',            # other plots are available
         data=df,
         order=2)

# creates a JOINTPLOT w/ SCATTER and MARGINAL KDEPLOT
g = (sns.jointplot(x="temp",
             y="registered",
             kind='scatter',
             data=df,
             marginal_kws=dict(bins=10, rug=True)).plot_joint(sns.kdeplot))

plt.show()
```

---

## Bokeh

---

Visual properties of shapes are called glyphs in Bokeh. The visual properties of these glyphs such as position or color can be assigned single values (size=10, fill_color='red'). Other glyph properties can be set as lists or arrays (x=[1,2,3], size=[10,20,30]).

### Dependencies

allows plots to be saved to an **html file** and using a browser to display the file

```python
from bokeh.io import output_file, show
```

allows plots to be displayed inline in a **jupyter notebook**

```python
from bokeh.io import output_notebook, show
```

imports the **figure object** to be able to create a figure instance

```python
from bokeh.plotting import figure
```

imports **Column Data Source object** to be able to create a ColumnDataSource instance

```python
from bokeh.models import ColumnDataSource
```

imports the **Hover Tool object** to be able to create a HoverTool instance

```python
from bokeh.models import HoverTool
```

imports the **Categorical Color Mapper object** to be able to create a CategoricalColorMapper instance

```python
from bokeh.models import CategoricalColorMapper
```

imports the **row method** to place glyphs side by side

```python
from bokeh.layouts import row
```

imports the **column method** to place glyphs on top of one another

```python
from bokeh.layouts import column
```

imports the **Grid Plot method** which allows to create rows and columns in one method

```python
from bokeh.layouts import gridplot
```

imports **Tabs and Panels** which allow creation of tabbed layouts

```python
from bokeh.models.widgets import Tabs, Panel
```

### Exporting

saves the plot to an **html file**

```python
output_file('file_name.html')
```

### Graphing

creates the **figure object** to be used; plot is commonly referenced as 'p'

```python
# can also use 'plot_height' parameter; can utilize other 'tool' objects (box_select, lasso_select, etc.)
plot = figure(plot_width=400, tools='pan, box_zoom')
```

**marker glyphs** determine the mark style used on the plot

```python
# can use other markers such as asterisk(), cross(), square(), triangle(), etc.
plot.circle(x, y)ls

# displays the plot in a browser(HTML) or inline a Jupyter Notebook
show(plot)
```

**line glyphs** plots a line through the coordinates specified

```python
# can use other parameters such as 'line_width'
 plot.line(x, y)

show(plot)
```

### Modifier Commands

creates a **ColumnDataSource instance** which defines data that can be used on multiple glyphs in a plot

```python
# source is created by passing a data dictonary through the ColumnDataSource initializer
# all columnns in the column data source instance must be the same length
source = ColumnDataSource( data={
    'x': [1,2,3,4,5],
    'y':[6,7,8,9,10]
})

# can also create a column data source instance using DataFrames
source = ColumnDataSource(df)
```

creates a **HoverTool instance** which defines the hover behavior of the plot

```python
hover = HoverTool(tooltips=None, mode='hline')

# passing the hover instance in the 'tool' parameter to be able to specify a hover policy in the 'plot' figure object
plot = figure(tools=[hover, 'crosshair'])

# if the plot has been defined, use the 'add_tools()' function to add more tools
p.add_tools(hover)

# a circle glyph with hover policies
plot.circle(x, y, hover_color='red', hover_size=10)
```

creates a **CategoricalColorMapper instance** which defines the colors to be mapped to the data

```python
mapper = CategoricalColorMapper(
    factors=['Data_Category_1', 'Data_Category_2', 'Data_Category_3'],
    palette=['red', 'green', 'blue']
)

# a circle glyph with color mapping policies by passing a 'color' dictionary
# the 'field' value should be the column containing the 'factors'
plot.circle(x, y, color={'field': 'Column_Name',
                         'transform':mapper})

# can also be formatted as follows
plot.circle(x, y, color=dict(field='origin', transform=mapper))
```
