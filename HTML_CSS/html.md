# HTML <!-- omit in toc -->

# Table of Contents <!-- omit in toc -->

- [Content Categories for Elements](#content-categories-for-elements)
  - [Flow Content](#flow-content)
  - [Sectioning Content](#sectioning-content)
  - [Heading Content](#heading-content)
  - [Phrasing Content](#phrasing-content)
- [Elements](#elements)
- [Attributes](#attributes)

# Content Categories for Elements

## Flow Content

- generally contain text or embedded content such as images and videos.

- `<header>` element
  - represents introductory content, typically a group of introductory or navigational aids.
- `<main>` element
  - represents the main content of the document, content which is unique to this particular page.
- `<footer>` element
  - section things like social media links, secondary navigation, and site small print.

## Sectioning Content

- includes special container elements used for grouping other elements into meaningful collections.

  - ie. `<article>`, `<section>`, `<nav>`
  - note this does not include the `<header>`, `<main>`, and `<footer>` elements

- `<section>` element
  - represent a thematically related group of elements which forma a component of a larger whole.
  - ie. website with area for blog posts and area for merchandise links.
- `<article>` element
  - used to mark a self-contained composition w/in a document; element which makes sense on its own as a standalone work.
  - ie. weather widget w/each daily forecast a separate article.
- `<div>` element
  - very generic by design; don't carry inherent meaning which makes them useful as generic grouping elements.
  - primarily used to aid in styling a page or to add behavior to some section of the browser window.
- `<nav>` element
  - major navigation blocks within the site;
    - ie. header bar with hyperlinks, table of contents, and menus.

## Heading Content

- reserved for the six heading elements in HTML; can only contain phrasing content.
  - ie. `<h1>`, `<h2>`, `<h3>`, etc.

## Phrasing Content

- represents the text of the document and the elements which mark up that text.
  - also includes `<img>`, `<audio>`, and `<video>` as these elements revert to plain text if the relevant resource can't be retrieved.

# Elements

|  Command  |                                                                      |
| :-------: | -------------------------------------------------------------------- |
| `<title>` | creates the title element which is displayed in the browser toolbar. |
|   `<h>`   | creates header element defined `1-6` for sizes.                      |
|   `<p>`   | creates paragraph element.                                           |
| `<span>`  | annotates small pieces of text; very generic.                        |
|  `<ol>`   | ordered list.                                                        |
|  `<ul>`   | unordered list.                                                      |
|  `<li>`   | list elements in ordered/unordered list.                             |
|  `<em>`   | denotes stress emphasis in text.                                     |

image `<img>` elements are used to display images.

```html
<!-- alt provides context for when the image does not load -->
<img src="http://www.website.jpg" alt="placeholder-for-Image" />
```

anchor `<a>` element creates a link to a new webpage using `href`.

```html
<!-- LINK w/NEW TAB -->
<a href="https://www.github.com">This is the Text for the Link</a>

<!-- THIS IS A PLACEHOLDER LINK -->
<a href="#">This is a Placeholder.</a>
```

form `<form>` element is used to retrieve information from the user

```html
<!-- action = where to send form data when form is submitted -->
<!-- method = POST allows form data to be submitted -->
<form action="/login" method="post">
  <h1>Form Header</h1>
  <p>You would have form information and data to go here</p>
</form>
```

script `<script>` element is used to connect other files to the html document.

```html
<script type="text/javascript" src="app.js"></script>
<!-- src is "source" of js file -->
```

# Attributes

- `<class>` attribute
  - "global" attribute; can be applied to any HTML element.
  - used to identify particular elements when applying styles, or when selecting a certain group of elements is needed.

```html
<!-- p element w/multiple classes -->
<p class="class-one class-two class-three">That's a lot of classes!</p>
```

- `<id>` attribute
  - used to identify particular elements when working with JS, to implement behavior for a particular element.
  - unlike classes, ids are supposed to be unique identifiers.

```html
<h1 class="title" id="pageTitle">Awesome Page Title</h1>
```
