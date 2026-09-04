// ============================================================
// Trip Planner Graph Schema - Constraints & Indexes
// Run once against a fresh Neo4j database (5.x, APOC installed)
// ============================================================

// --- Core domain entity ---
CREATE CONSTRAINT destination_id IF NOT EXISTS
FOR (d:Destination) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT osm_feature_id IF NOT EXISTS
FOR (f:OsmFeature) REQUIRE f.id IS UNIQUE;

// --- Lookup / dimension nodes (deduplicated via MERGE) ---
CREATE CONSTRAINT province_name IF NOT EXISTS
FOR (p:Province) REQUIRE p.name IS UNIQUE;

CREATE CONSTRAINT city_name IF NOT EXISTS
FOR (c:City) REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT category_name IF NOT EXISTS
FOR (c:Category) REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT season_name IF NOT EXISTS
FOR (s:Season) REQUIRE s.name IS UNIQUE;

CREATE CONSTRAINT triptype_name IF NOT EXISTS
FOR (t:TripType) REQUIRE t.name IS UNIQUE;

CREATE CONSTRAINT vehicle_name IF NOT EXISTS
FOR (v:Vehicle) REQUIRE v.name IS UNIQUE;

CREATE CONSTRAINT exitroute_name IF NOT EXISTS
FOR (e:ExitRoute) REQUIRE e.name IS UNIQUE;

CREATE CONSTRAINT landmark_name IF NOT EXISTS
FOR (l:Landmark) REQUIRE l.name IS UNIQUE;

CREATE CONSTRAINT suitability_name IF NOT EXISTS
FOR (s:Suitability) REQUIRE s.name IS UNIQUE;

CREATE CONSTRAINT osm_feature_type_id IF NOT EXISTS
FOR (t:OsmFeatureType) REQUIRE t.id IS UNIQUE;

CREATE CONSTRAINT country_code IF NOT EXISTS
FOR (c:Country) REQUIRE c.code IS UNIQUE;

// --- Agent / user memory layer (grows through interaction) ---
// NOTE: no constraint on User here. neo4j-agent-memory's SchemaManager
// already creates `CREATE CONSTRAINT ... FOR (u:User) REQUIRE u.identifier
// IS UNIQUE` automatically on every MemoryClient.connect() -- i.e. every
// service startup. TripGraphRepository (src/memory/graph.py) now keys its
// own User writes on `identifier` too, specifically so they land on that
// same constrained node instead of creating a second, disconnected one.
// Declaring a second constraint on `id` here (as this file used to) would
// protect the wrong property and re-introduce the split.

CREATE CONSTRAINT trip_id IF NOT EXISTS
FOR (t:Trip) REQUIRE t.id IS UNIQUE;

CREATE CONSTRAINT travel_goal_session IF NOT EXISTS
FOR (g:TravelGoal) REQUIRE g.session_id IS UNIQUE;

// --- Vector index for semantic search over destination descriptions ---
// Dimension must match your embedding config (settings.embedding_dimensions
// in src/config.py). Set here to 768 to match AvalAI's sample request
// (text-embedding-3-small with dimensions=768, not the model's 1536 default).
// If you later change embedding_dimensions, drop and recreate this index --
// a mismatch is not auto-detected the way it is for neo4j-agent-memory's own
// managed indexes (message/entity/preference/... via EmbeddingDimensionMismatchError).
CREATE VECTOR INDEX destination_embedding IF NOT EXISTS
FOR (d:Destination) ON (d.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}};

// --- Geo / text indexes ---
CREATE POINT INDEX destination_location IF NOT EXISTS
FOR (d:Destination) ON (d.location);

CREATE FULLTEXT INDEX destination_fulltext IF NOT EXISTS
FOR (d:Destination) ON EACH [d.name, d.description];
