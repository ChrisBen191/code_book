# Neo4j Knowledge <!-- omit in toc -->

Knowledge on Neo4j, the native graph database utilizing index-free adjacency (IFA).

# Table of Contents <!-- omit in toc -->

- [Graph Data Modeling](#graph-data-modeling)
- [Index-Free Adjacency (IFA)](#index-free-adjacency-ifa)
- [Definitions](#definitions)
- [Neo4j Browser Commands](#neo4j-browser-commands)
- [Cypher Syntax](#cypher-syntax)
- [Creating / Updating Nodes](#creating--updating-nodes)
- [Creating / Updating Relationships](#creating--updating-relationships)
- [Common Cypher Queries](#common-cypher-queries)
- [Merging Data in the Graph](#merging-data-in-the-graph)
- [Using Indexes](#using-indexes)
  - [Constraints](#constraints)
  - [Node Keys](#node-keys)
  - [Indexes](#indexes)
  - [Full-Text Schema Indexes](#full-text-schema-indexes)
- [Query Best Practices](#query-best-practices)
- [Importing Data](#importing-data)
  - [CSV](#csv)
- [APOC](#apoc)

# Graph Data Modeling
Two types of models are required when preforming the graph data modeling process for an application:
    1. Data model - describes the labels, relationships, and properties for the graph; does not have specific data.
    2. Instance model - set of sample data to test the data model against any number of use cases.

# Index-Free Adjacency (IFA)

Utilized in graph databases; relationships are stored at write-time in a graph database, meaning the query time will remain consistent to the size of the data that is actually touched during a query.

# Definitions

| Definition             | Meaning                                                                                  |
| :--------------------- | :--------------------------------------------------------------------------------------- |
| node                   | vertices in a data graph; represent objects, entities, or things.                        |
| node label             | identifies the subset a node belongs to.                                                 |
| node property          | key, value pair of info providing additional context about the node.                     |
| relationship           | line or edge in a data graph; describes how nodes are connected together.                |
| relationship direction | identifies the direction of connection between nodes, important in context of hierarchy. |
| relationship type      | identifies which part of the graph to traverse.                                          |
| relationship property  | key, value pair of info providing additional context about the relationship.             |
| undirected graph       | graph with bi-directional or symmetric relationships.                                    |
| directed graph         | graph with single directional or asymmetrical relationships.                             |
| weighted graph         | graph with relationships that carry a value tha represents a weight or measure.          |
| unweighted graph       | graph with relationships that do not carry a measurable value.                           |
| graph traversal        | the path or relationships traveled to obtain the graph.                                  |

# Neo4j Browser Commands

| Command                                       | Definition                                                                                                             |
| --------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| `:clear`                                      | removes all frames from the stream.                                                                                    |
| `:config`                                     | displays configuration settings.                                                                                       |
| `:history`                                    | displays history of executed commands.                                                                                 |
| `:schema`                                     | displays info about the database schema indexes and constraints.                                                       |
| `:sysinfo`                                    | displays info about store size, id allocation, page cache, etc.                                                        |
| `CALL db.schema.visualization()`              | Visualizes the data model of the graph, to better understand the nodes, labels, and relationships of the graph.        |
| `CALL db.constraints()` or `CALL CONSTRAINTS` | Displays the set of constraints that have been defined for the graph.                                                  |
| `CALL db.indexes()` or `SHOW INDEXES`         | Displays a list of all full-text schema indexes for the graph.                                                         |
| `CALL gds.graph.list()`                       | Displays all created named graphs and their related graph information.                                                 |
| `PROFILE MATCH...`                            | Runs the Cypher query statement, and provides run-time performance metrics.                                            |
| `EXPLAIN MATCH...`                            | Provides the Cypher query plan; provides estimates of the graph engine processing that will occur (doesn't run query). |

# Cypher Syntax

|            Node Syntax            | Meaning                                                                                   |
| :-------------------------------: | :---------------------------------------------------------------------------------------- |
|           `()` or `(n)`           | use (n) to denote an "anonymous" node for further query processing in the query.          |
| `(n:NodeLabel)` or `(:NodeLabel)` | node labels can also be used with anonymous nodes and more than one label can be denoted. |
|             `()--()`              | 2 nodes and any symmetrical relationship.                                                 |
|   `()-[:RELATIONSHIP_TYPE]-()`    | 2 nodes and a specific symmetrical relationship type.                                     |
|      `(first)-[]->(second)`       | the `first` node with any relationship type to the `second` node.                         |
|      `(first)<-[]-(second)`       | the `second` node with any relationship type to the `first` node.                         |

# Creating / Updating Nodes

Create a node using `CREATE`; this method does not look up the primary key before adding the node and can create duplicates.
```cypher
CREATE (nodeVariable: NodeLabels {optionalProperties})

// adding a 'hero' node to the graph w/name property
CREATE (h:Hero {name: 'Peter Parker'})
```

Create a node using `MERGE`; this method looks up the primary key before adding the node, therefore avoiding duplication.
```cypher
// adding a 'hero' node to the graph if not existing 
MERGE (h:Hero {name: 'Peter Parker'})
```

When defining label for a node, dominant nouns are assigned to the entities in the graph (ie. ingredient, movie, person, etc.).
Node labels serve as an anchor point for a query; using a label helps reduce the amount of data that is retrieved. 


Add/Remove node labels using `SET` and `REMOVE`
```cypher
MATCH (h:Hero)
WHERE h.name = 'Peter Parker'

// adding label to the hero node if not already existing
SET h:Avenger, h:Amateur

// removing label from the hero node
REMOVE h:Amateur

// returns labels associated w/the node
RETURN labels(h)
```

Add node label based on a relationship.
```cypher
MATCH (h:Hero)
WHERE exists ((h)-[:WON_AGAINST]-())

// adding label to the hero node w/WON_AGAINST relationship
SET h:Champion
```

Properties for a node uniquely identify a node, provide flags, and return data.

Add/Remove node properties using `SET` and `REMOVE`
```cypher
MATCH (h:Hero)
WHERE h.name = 'Peter Parker'

// adding one property to hero node
SET h.alias = 'Spiderman'

// this method must include all properties/values as all existing properties are overwritten
SET h = {name: 'Peter Parker', alias: 'Spiderman', employment: 'Unemployed'}

// this method will update existing and add properties not existing
SET h += {employment: 'Daily Bugle', symbiote: False}

// removing property from the hero node
REMOVE m.symbiote

// removing property from the hero node by setting to NULL
SET h.employment = null

RETURN h
```

Delete a node using `DELETE`; this method is successful provided no relationships w/the node exist.
```cypher
MATCH (h:Hero)
WHERE h.name = 'Peter Parker'
DELETE h
```

Delete a node using `DETACH DELETE`; this method removes a node w/existing relationships.
```cypher
MATCH (h:Hero)
WHERE h.name = 'Peter Parker'
DETACH DELETE  h
```

# Creating / Updating Relationships
Relationships are the connection between entities or nodes. When defining a relationship type, verbs are assigned to the connections in the 
graph. (ie. USES, MARRIED_TO, FOUGHT_AGAINST).

A direction must either be specified explicitly, or inferred by the left-to-right direction in the pattern specified.

Fanout occurs when entities are spread out and represented as a network or linked nodes. Fanout can lead to supernodes, or very dense nodes;
the splitting up of nodes should be done only to answer questions or use cases. 

Create a relationship using `CREATE`; this method does not account for existing relationships and can create duplicates.
```cypher
MATCH (h:Hero), (v:Villain)
WHERE h.name = 'Peter Parker'
    AND v.alias = 'Green Goblin'

// creates the relationship between nodes
CREATE (h)-[:FOUGHT_AGAINST]->(v)

RETURN h, v
```

Create a relationship using `MERGE`; this method accounts for existing relationships, therefore avoiding duplication.
```cypher
MATCH (h:Hero), (v:Villain), (s:Sidekick)
WHERE h.hame = 'Peter Parker'
    AND v.alias = 'Green Goblin'
    AND s.alias = 'Shockwave'

// creates multiple relationships to the nodes, if not existing
MERGE (h)-[:FOUGHT_AGAINST]->(v)<-[:SIDEKICK_OF]-(s)

RETURN h, v, s
```

Add/Remove relationship properties using `SET` and `REMOVE`
```cypher
MATCH (h:Hero)-[rel:FOUGHT_AGAINST]->(v:Villain)

WHERE h.name = 'Peter Parker'
    AND v.alias = 'Green Goblin'

// adding location and times properties for relationship
SET 
    rel.location = ['Times Square', 'MSG', 'Avenger Tower'], 
    rel.times = [2002, 2006, 2010]

// removing property from relationship
REMOVE rel.times

// removing property from relationship by setting to NULL
rel.location = null

RETURN h, rel, v
```

# Common Cypher Queries

| Command                                                                      | Definition                                                                                   |
| ---------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------- |
| `CALL db.propertyKeys()`                                                     | Displays all properties of a graph; does not what define which each property key belongs to. |
| `MATCH (n:Node) RETURN keys(n)`                                              | Displays the properties of a node.                                                           |
| `:params {paramName: 'paramValue, ...'}` OR `:param paramName => paramValue` | Sets defined parameter and its value in the current session.                                 |
| `:params`                                                                    | Displays the current parameters in session and their values                                  |
| `:queries`                                                                   | Displays current running queries to allow for monitoring/troubleshooting                     |
| `CALL gds.graph.drop(<graph-name>)`                                          | Drops the named graph specified.                                                             |


Create/Update properties depending on if a node exists or not using `ON CREATE SET` and `ON MATCH SET`
```cypher
MERGE (h:Hero {name: 'Peter Parker'})

// property is created if node is created
ON CREATE SET h.alias = 'Spiderman'

// property is updated if node is existing
ON MATCH SET h.employed = 'Daily Bugle'

// property is set regardless
SET h.loveInterest = 'Gwen Stacy'

RETURN h
```

Creating a **MATCH** clause that includes a pattern specified by two paths.
```cypher
// returning actors and director from movie(s) released in 2000.
MATCH (a:Person)-[:ACTED_IN]->(m:Movie), (m)<-[:DIRECTED]-(d:Person)
WHERE m.released = 2000
Return a.name as ACTOR, m.title AS TITLE, d.name AS DIRECTOR

// when multi patterns are specified in a MATCH clause, no relationship is traversed more than one time
MATCH (a:Person)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Person)
WHERE m.released = 2000
Return a.name as ACTOR, m.title AS TITLE, d.name AS DIRECTOR
```

Using two patterns in a **MATCH** statement

```cypher
// matching keanu reeves' movies and hugo weaving node
MATCH (keanu:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(n:Person), (hugo:Person)
WHERE keanu.name = 'Keanu Reeves'
    AND hugo.name = 'Hugo Weaving'
    AND NOT (hugo)-[:ACTED_IN]->(m) // filtering for movies w/out hugo weaving
RETURN n.name AS actor
```

Traversal using multiple patterns in a **MATCH** clause

```cypher
MATCH (valKilmer:Person)-[:ACTED_IN]->(m:Movie),    //pattern retrieves TopGun node
      (actor:Person)-[:ACTED_IN]->(m)               // pattern retrieves TopGun actor nodes
WHERE valKilmer.name = 'Val Kilmer'
RETURN m.title as movie , actor.name

// the result does not include the Val Kilmer node
```

**MATCH** clause that defines the number of hops for relationship on a path

```cypher
// retrieves all person nodes that are two hops away
MATCH (follower:Person)-[:FOLLOWS*2]->(p:Person)
WHERE follower.name = 'Paul Blythe'
RETURN p.name

// using [:FOLLOWS*] would return all Person nodes that are in the :FOLLOWS path from 'Paul Blythe'
```

The **shortestPath()** function finds the shortest path between nodes to improve the query.

```cypher
// specify variable p as result of calling shortestPath() specify * for any relationship between nodes
MATCH p = shortestPath((m1:Movie)-[*]-(m2:Movie))
WHERE m1.title = 'A Few Good Men' AND
      m2.title = 'The Matrix'
RETURN  p
```

Creating a **subgraph**, or a set of paths derived from the **MATCH** clause

```cypher
MATCH paths = (m:Movie)--(p:Person)
WHERE m.title = 'The Replacements'
RETURN paths
```

**OPTIONAL MATCH** (similar to SQL outer join) returns Nulls for missing parts of the pattern

```cypher
MATCH (p:Person)
WHERE p.name STARTS WITH 'James'
OPTIONAL MATCH (p)-[r:REVIEWED]->(m:Movie) // if 'James' is an actor, null will be returned
RETURN p.name, type(r), m.title
```

Common way to aggregate data using **count()** function

```cypher
// counts # of movies where actor and director included in same movie
MATCH (a:Person)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Person)
RETURN a.name, d.name, count(m)
```

The **collect()** method will aggregate a value into a list.

```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE p.name = 'Tom Cruise'
RETURN collect(m.title) AS `movies for Tom Cruise` // movies are returned as a list of values

// The collect() method can also be used to collect nodes
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE p.name = 'Tom Cruise'
RETURN collect(m) AS `movies for Tom Cruise` // movies returned as nodes
```

Counting the **paths** found for each actor/director collaboration and returning movies as list.

```cypher
MATCH (actor:Person)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(director:Person)
RETURN
    actor.name,
    director.name,
    count(m) AS collaborations, // count of collaboration
    collect(m.title) AS movies  // list of movies for each actor/director collaboration
```

Using a **map projection** when retrieving nodes to return some of the information and not all of it.

```cypher
MATCH (m:Movie)
WHERE m.title CONTAINS 'Matrix'
RETURN m { .title, .released } AS movie // tagline property is not included
```

The **date()** and **datetime()** functions store their values as strings, so date properties can be extracted.

```cypher
// returning properties of today's date; day, year, hour, and minute values
RETURN date().day, date().year, datetime().year, datetime().hour, datetime().minute
```

The **timestamp()** function stores its value as a long integer, requiring conversion before values can be extracted.

```cypher
// returning the day, year, and, month values from the timestamp value.
RETURN datetime({epochmillis:timestamp()}).day,
       datetime({epochmillis:timestamp()}).year,
       datetime({epochmillis:timestamp()}).month
```

Using the **WITH** clause allows you to specify intermediate calculations or values that will be used further in the query.

```cypher
MATCH (a:Person)-[:ACTED_IN]->(m:Movie)
WITH  a, count(m) AS numMovies, collect(m.title) as movies // defined variables in WITH clause can be used later in query
WHERE 1 < numMovies < 4
RETURN a.name, numMovies, movies
```

Using the **UNWIND** clause will provide nodes collected with collect() function to be displayed in rows

```cypher
MATCH (m:Movie)<-[:ACTED_IN]-(p:Person)
WITH collect(p) AS actors, count(p) AS actorCount, m
UNWIND actors AS actor      // "unwinding" the names of actors that were collected
RETURN m.title, actorCount, actor.name
```

**Subqueries** can be produced using the **WITH** clause, so the second query can access the properties required

```cypher
// first match clause finds all reviewers of movies
MATCH (m:Movie)<-[rv:REVIEWED]-(r:Person)
WITH m, rv, r

// second match clause finds directors of movies
MATCH (m)<-[:DIRECTED]-(d:Person)

// returns movie, reviewer rating, reviewer, and names of directors
RETURN m.title, rv.rating, r.name, collect(d.name)
```

**Subquery** with the **OPTIONAL MATCH** clause will provide additional data from second query if data is available; similar to SQL outer join.

```cypher
MATCH (p:Person)
WITH p, size((p)-[:ACTED_IN]->()) AS movies
WHERE movies >= 5

// optionally will return values if p (actor) is also found to have directed, else return 'null'
OPTIONAL MATCH (p)-[:DIRECTED]->(m:Movie)
RETURN p.name, m.title
```

Subqueries can also be achieved using the **CALL** function with **{ }** around subquery

```cypher
// first query collects movies that have been reviewed
CALL {
    MATCH (p:Person)-[:REVIEWED]->(m:Movie)
    RETURN m
}
// second query shows movies (that were reviewed) released in 2000
MATCH (m)
WHERE m.released=2000
RETURN m.title, m.released
```

---


# Merging Data in the Graph

Use the **MERGE** clause to create/update a **node**; does take into account if the node is existing.

```cypher
// generic framework
MERGE (variable:Label{nodeProperties})
RETURN variable

MERGE (a:Actor {name: 'Michael Caine'})  // creates new Actor node if not existing
SET a.born = 1933
RETURN a
```

Use the **MERGE** clause to create / update a **relationship** as well.

```cypher
// generic framework
MERGE (variable1:Label1 {nodeProperties1})-[:REL_TYPE]->(variable2:Label2 {nodeProperties2})
RETURN variable1, variable2


MATCH
    (p:Person {name: 'Michael Caine'}),
    (m:Movie {title:'Batman Begins'})

MERGE (p)-[:ACTED_IN]->(m)          // creates 'ACTED_IN' relationship between 'michael caine' and 'batman begins' if not existing
RETURN p,m
```

Specify the creation behavior when merging by using the **ON CREATE SET / ON MATCH SET** clauses.

```cypher
MERGE (a:Person {name: 'Sir Michael Caine'})
ON CREATE SET
            a.birthPlace = 'London',      // if node doesn't exist, it will create a new node
            a.born = 1934                 // with defined 'birthPlace', 'born' node properties
ON MATCH SET
            a.birthPlace = 'UK'            // if node does exist, it will set/update the 'birthPlace' node property
RETURN a
```

Relationships can be created using the **MERGE** clause; can be expensive as this has the potential of creating two new nodes and relationship.

```cypher
MATCH (p:Person), (m:Movie)
WHERE m.title = 'Batman Begins' AND p.name ENDS WITH 'Caine'
MERGE (p)-[:ACTED_IN]->(m)              // if relationship doesn't exist, it will create it
RETURN p, m
```

---

# Using Indexes

## Constraints

Add a **uniqueness constraint** to the graph that asserts that a particular node property is unique for that node type. Uniqueness constraints can be only created for nodes.

```cypher
// this will prevent another Node being created with the same 'title' node property
CREATE CONSTRAINT UniqueConstraintName ON (v:NodeVariable) ASSERT v.title IS UNIQUE
```

Add an **existence constraint** to the graph that asserts that a particular type of node or relationship property must exist in the graph when a node/relationship of that type is created/updated.

```cypher
// this will prevent a Node being created without the 'tagline' node property being defined
CREATE CONSTRAINT ExistsConstraintName ON (v:NodeVariable) ASSERT exists(v.tagline)
```

An **existence constraint** can also be used with relationship properties.

```cypher
// this will prevent a 'REVIEWED' relationship from being created without the 'rating' relationship property being defined
CREATE CONSTRAINT ExistsREVIEWEDRating
ON ()-[rel:REVIEWED]-() ASSERT exists(rel.rating) // defining relationship pattern, asserting rating exists
```

Both a **uniqueness constraint** and an **existence constraint** can be removed using the **DROP** clause.

```cypher
DROP CONSTRAINT ExistingConstraintName
```

## Node Keys

A **node key** is used to define the uniqueness and existence constraint for multiple properties of a node of a certain type. A node key is also used as a composite index in the graph.

```cypher
CREATE CONSTRAINT UniqueNameBornConstraint
ON (p:Person) ASSERT (p.name, p.born) IS NODE KEY  // asserting both name and born exists
```

## Indexes

**Indexes** are used to improve initial node lookup performance, but they require additional storage in the graph to maintain and also add to the cost of creating or modifying property values that are indexed. **Indexes** store redundant data that points to nodes with the specific property value or values.

Create a **single-property index**; used for equality checks, range comparisons, listing membership, string comparisons, etc.

```cypher
CREATE INDEX MovieReleased FOR (m:Movie) ON (m.released)
```

Create a **composite index**; used for when there can be duplication for a set of property values and you want faster access to them.

```cypher
CREATE INDEX MovieReleasedVideoFormat FOR (m:Movie) ON (m.released, m.videoFormat)
```

Both a **single-property index** and a **composite index** can be removed using the DROP clause.

```cypher
DROP INDEX MovieReleasedVideoFormat
```

## Full-Text Schema Indexes

A **full-text schema index** is based upon string values only, and can be used for:

- node/relationship properties
- single/multiple properties
- single/multiple types of nodes(labels)
- single/multiple types of relationships

Create a **full-text schema index** for **nodes** by using the **createNodeIndex()** function.

```cypher
CALL db.index.fulltext.createNodeIndex(
      'MovieTitlePersonName',           // name of full-text schema index
      ['Movie', 'Person'],              // nodes to be indexed
      ['title', 'name'])                // node properties to be indexed
```

Call the query procedure using the **full-text schema index** for **nodes** by using the **queryNodes()** function.

```cypher
CALL db.index.fulltext.queryNodes('MovieTitlePersonName', 'Jerry')
YIELD node          // YIELD allows access to the property values in the nodes returned
RETURN node         // returns nodes in the graph with either a title property or name property containing 'Jerry'
```

Use the **AND , OR** clauses to query multiple strings in the **full-text schema index**.

```cypher
CALL db.index.fulltext.queryNodes('MovieTitlePersonName', 'Jerry OR Matrix')    // will find titles/names with either 'Jerry' or 'Matrix' in them
YIELD node          // YIELD allows access to the property values in the nodes returned
RETURN node         // returns nodes in the graph with either a title property or name property containing 'Jerry'
```

You can specify which property you want to search for in the defined **full-text schema index**.

```cypher
CALL db.index.fulltext.queryNodes('MovieTitlePersonName', 'name: Jerry')
YIELD node          // returns nodes in the graph with ONLY the name property containing 'Jerry'
RETURN node
```

You can retrieve a **"hit score"** that represents the closeness of the values in the graph to the query string.

```cypher
CALL db.index.fulltext.queryNodes('MovieTitlePersonName', 'Matrix')
YIELD node, score      // returns a Lucene (index provider) score based upon how much 'Matrix' was part of the title property
RETURN node.title, score
```

Create a **full-text schema index** for **relationships** by using the **createRelationshipIndex()** function.

```cypher
// creating a full-text schema index for relationship
CALL db.index.fulltext.createRelationshipIndex(
    'ActedRoleWriterName',              // name of full-text schema index
    ['ACTED', 'WROTE'],                 // relationships to be indexed
    ['role', 'pseudonym'])              // relationship properties to be indexed
```

You can drop a **full-text schema index** by using the **DROP** clause.

```cypher
CALL db.index.fulltext.drop('MovieTitlePersonName')
```

---

# Query Best Practices

Use **parameters** (denoted by $) in your queries as a change to a Cypher statement requires recompilation of the code which can be expensive

```cypher
:param actorName => 'Tom Hanks'
```

```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE p.name = $actorName
RETURN m.released, m.title ORDER BY m.released DESC
```

You can set **multiple parameters** to be used in a query.

```cypher
:params {actorName: 'Tom Cruise', movieName: 'Top Gun'}     // JSON style syntax for multiple parameters is also allowed
```

```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE p.name = $actorName AND m.title = $movieName
RETURN p, m
```

---

# Importing Data

## CSV

View the first 10 lines of a file/URL, with headers included and ',' as the delimeter

```cypher
LOAD CSV WITH HEADERS
FROM 'https://data.neo4j.com/v4.0-intro-neo4j/people.csv'
AS line
RETURN line LIMIT 10
```

You can filter the data prior to import using conditionals

```cypher
LOAD CSV WITH HEADERS
FROM 'https://data.neo4j.com/v4.0-intro-neo4j/people.csv'
AS line
WITH line WHERE line.birthYear > "1999"         // filters csv for rows where conditional is met
RETURN line LIMIT 10
```

Transform field values as needed as LOAD CSV as data is default interpreted as a string or null.

```cypher
LOAD CSV WITH HEADERS
FROM 'https://data.neo4j.com/v4.0-intro-neo4j/movies1.csv'
AS line
RETURN
    toFloat(line.avgVote),          // converts field values to FLOAT
    line.genres,
    toInteger(line.movieId),        // converts field values to INTEGER
    line.title,
    toInteger(line.releaseYear)     // converts field values to INTEGER
LIMIT 10
```

You can split lists (delimited with **:**) using the **split()** and **coalesce()** methods

```cypher
LOAD CSV WITH HEADERS
FROM 'https://data.neo4j.com/v4.0-intro-neo4j/movies1.csv'
AS line
RETURN
    toFloat(line.avgVote),
    split(coalesce(line.genres,""), ":"),   // if some fields have an empty list, use split() with coalesce()
    toInteger(line.movieId),
    line.title,
    toInteger(line.releaseYear)
LIMIT 10
```

Use **auto: USING PERIODIC COMMIT N** clause to import data over 100k rows; affected by "eager" operators (ie. collect(), count(), ORDER BY, DISTINCT)

```cypher
:auto USING PERIODIC COMMIT 500                                 // 500 rows will be uploaded at one time
LOAD CSV WITH HEADERS FROM
  'https://data.neo4j.com/v4.0-intro-neo4j/movies1.csv' as row
MERGE (m:Movie {id:toInteger(row.movieId)})
    ON CREATE SET                                               // assigning the string csv data to properties of Movie node
          m.title = row.title,
          m.avgVote = toFloat(row.avgVote),
          m.releaseYear = toInteger(row.releaseYear),
          m.genres = split(row.genres,":")
```

Importing **relationships** related to data already imported (Movie data)

```cypher
LOAD CSV WITH HEADERS FROM
'https://data.neo4j.com/v4.0-intro-neo4j/directors.csv' AS row
MATCH (movie:Movie {id:toInteger(row.movieId)})         // for each row read, find the Movie node
MATCH (person:Person {id: toInteger(row.personId)})     // for each row read, find the Person node
MERGE (person)-[:DIRECTED]->(movie)                     // create DIRECTED relationship between nodes
ON CREATE SET person:Director                           // add DIRECTOR label to Person node
```

# APOC

|                   Syntax | Meaning                                                                                                    |
| -----------------------: | :--------------------------------------------------------------------------------------------------------- |
| `CALL apoc.meta.graph()` | Retrieves high-level metadata from teh graph to inspect how nodes and relationships are used in the graph  |
| `CALL apoc.meta.stats()` | Retrieves count store data which provides information about the number of nodes/relationships of each type |

Use APOC clear the database of all **constraints**, **indexes**, **nodes**, and **relationships**; helpful when preforming multiple imports to database during testing.

```cypher
CALL apoc.schema.assert( {}, {}, true);     // removes all constraints and indexes

CALL apoc.periodic.iterate(                 // removes all nodes and relationships
    'MATCH (n) RETURN n',
    'DETACH DELETE n',
    {batchSize: 500}
)
```

Use APOC to create a **conditional** when uploading **denormalized data** using the **apoc.do.when()** procedure

```cypher
CREATE CONSTRAINT UniqueMovieIdConstraint ON (m:Movie) ASSERT m.id IS UNIQUE;
CREATE CONSTRAINT UniquePersonIdConstraint ON (p:Person) ASSERT p.id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'https://data.neo4j.com/v4.0-intro-neo4j/movies2.csv' AS row
WITH
    row.movieId as movieId,                 // properties to be used for Movie node
    row.title AS title,
    row.genres AS genres,
    toInteger(row.releaseYear) AS releaseYear,
    toFloat(row.avgVote) AS avgVote,
    collect({                               // properties to be used for Person node; being collected to define later in the query
        id: row.personId,
        name:row.name,
        born: toInteger(row.birthYear),
        died: toInteger(row.deathYear),personType: row.personType,
        roles: split(coalesce(row.characters,""),':')}) AS people

MERGE (m:Movie {id:movieId})                // creating the Movie node, and defining its properties
   ON CREATE SET
            m.title=title,
            m.avgVote=avgVote,
            m.releaseYear=releaseYear,
            m.genres=split(genres,":")

WITH *                                      // carries all variables forward in the query
UNWIND people AS person                     // all "peope" collected unwinded as rows

MERGE (p:Person {id: person.id})            // creating the Person node, and defining its properties
   ON CREATE SET
            p.name = person.name,
            p.born = person.born,
            p.died = person.died

WITH  m, person, p
CALL apoc.do.when(person.personType = 'ACTOR',              // personType property used as conditional
     "MERGE (p)-[:ACTED_IN {roles: person.roles}]->(m)      // create 'ACTED_IN' relationship if personType == 'ACTOR'
                ON CREATE SET p:Actor",
     "MERGE (p)-[:DIRECTED]->(m)                            // create 'DIRECTED' relationship if personType != 'ACTOR'
         ON CREATE SET p:Director",
     {m:m, p:p, person:person}) YIELD value                 // describes mapping of variables outside/inside of the call; for simplicity the same
RETURN count(*)                                             // certain apoc calls cannot end a cypher query so count(*) is placed at the end
```

Use APOC to import data from a CSV that cannot be uploaded using the LOAD CSV function

```cypher
CALL apoc.periodic.iterate(
  "CALL apoc.load.csv('https://data.neo4j.com/v4.0-intro-neo4j/movies2.csv' ) YIELD map AS row
   RETURN row",                               // using apoc.load.csv() to upload data, returns as row
  "WITH
      row.movieId as movieId,                 // properties to be used for Movie node
      row.title AS title,
      row.genres AS genres,
      toInteger(row.releaseYear) AS releaseYear,
      toFloat(row.avgVote) AS avgVote,
      collect({                               // properties to be used for Person node; being collected to define later in the query
          id: row.personId,
          name:row.name,
          born: toInteger(row.birthYear),
          died: toInteger(row.deathYear),personType: row.personType,
          roles: split(coalesce(row.characters,""),':')}) AS people

  MERGE (m:Movie {id:movieId})                // creating the Movie node, and defining its properties
     ON CREATE SET
              m.title=title,
              m.avgVote=avgVote,
              m.releaseYear=releaseYear,
              m.genres=split(genres,":")

  WITH *                                      // carries all variables forward in the query
  UNWIND people AS person                     // all "peope" collected unwinded as rows

  MERGE (p:Person {id: person.id})            // creating the Person node, and defining its properties
     ON CREATE SET
              p.name = person.name,
              p.born = person.born,
              p.died = person.died

  WITH  m, person, p
  CALL apoc.do.when(person.personType = 'ACTOR',              // personType property used as conditional
       "MERGE (p)-[:ACTED_IN {roles: person.roles}]->(m)      // create 'ACTED_IN' relationship if personType == 'ACTOR'
                  ON CREATE SET p:Actor",
       "MERGE (p)-[:DIRECTED]->(m)                            // create 'DIRECTED' relationship if personType != 'ACTOR'
           ON CREATE SET p:Director",
       {m:m, p:p, person:person}) YIELD value                 // describes mapping of variables outside/inside of the call; for simplicity the same
  RETURN count(*)",
  {batchsize: 500}                                            // defines the size of the batch
)
```
