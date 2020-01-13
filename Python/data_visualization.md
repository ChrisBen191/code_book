# MATPLOTLIB
=========================

## DEPENDENCIES

imports the **matplotlib** dependency to create plots
```python
import matplotlib.pyplot as plt
```

## IMPORTING
loads **image** and displays it
```python
img = plt.imread('image_file.png')

# removes the axis on the image
plt.axis('off')

# displays the image
plt.imshow(img)
```

## EXPORTING
**exports** the plot
```python
# saving as a 'png' file
plt.savefig('image_name.png')
```

## SIZING ATTRIBUTES
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
## GRAPHING ATTRIBUTES
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
creates a **margin** around the plot to keep data off of the edges
```python
# the following provides a 2% buffer around plot edges
plt.margins(0.02)
```
## GRAPHING
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
=========================

## IMPORTS
imports the **seaborn** dependency to create plots with pandas DataFrames
```python
# seaborn is built on top of matplotlib; plt.show() displays seaborn plots
import seaborn as sns
```

## GRAPHING
plots a **linear regression** graph
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

plt.show()
```

plots the **residuals of a regression**; residual structure suggests a simple/single linear regression is not appropriate
```python
sns.residplot(x='Column_Name', y='Column_Name_2', data=df, color='red')

plt.show()
```

**strip plot** displays values on a number line to visualize samples of a single variable
```python
# using the 'x' parameter will provide a grouped strip plot
sns.stripplot(y='Column_Name', data=df)

# jitter spreads out split plot points (by size parameter) so they don't overlap
sns.stripplot(y='Column_Name', data=df, size=num, jitter=True)

plt.show()
```

**swarmplot** automatically arranges repeated points to avoid overlap and provide a sense of distribution
```python
sns.swarmplot(x='Column_Name', y='Column_Name_2', data=df)

# 'hue' parameter displays the colors of a categorial column within the plot
sns.swarmplot(x='Column_Name', y='Column_Name_2', data=df, hue='Column_Name_3')

plt.show()
```

**boxplot** displays the minimum, maximum, and median values of a dataset along the 1st and 3rd quartiles and outliers
```python
sns.boxplot(x='Column_Name', y='Column_Name_2', data=df)

plt.show()
```

**violinplots** show curved distributions (KDE) wrapped around a box plot 
```python
# the distribution is denser where the violin plot is thicker
sns.violinplot(x='Column_Name', y='Column_Name_2, data=df)

plt.show()
```

**jointplots** provide a plot with histograms (of individual coordinates) above and to the side of the main plot
```python
sns.jointplot(x='Column_Name', y='Column_Name', data=df)

plt.show()
```

**pairplots** provide jointplots for all possible pairs of numerical column variables in a df
```python
sns.pairplot(df)

plt.show()
```

**heatmaps** plot covariance matrices when pairplot() plots become visually overwhelming
```python
sns.heatmap(cov_matrix)

plt.show()
```

---

# BOKEH
=========================
Visual properties of shapes are called glyphs in Bokeh. The visual properties of these glyphs such as position or color can be assigned single values (size=10, fill_color='red'). Other glyph properties can be set as lists or arrays (x=[1,2,3], size=[10,20,30]).

## DEPENDENCIES

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

## EXPORTING

saves the plot to an **html file**
```python
output_file('file_name.html')
```

## GRAPHING

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

## MODIFIER COMMANDS

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

