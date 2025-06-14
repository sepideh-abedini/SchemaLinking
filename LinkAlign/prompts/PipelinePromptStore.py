"""  在此处定义所有的 Single-Prompt Pipeline 策略使用的提示词 """

# Prompt For Step One
QUERY_REWRITING_TEMPLATE = """You are a database semantic architect. Given a user question and partial schema context, systematically perform schema adequacy analysis and question re-engineering through these phases:
# Phase 1: Structural Necessity Analysis
A) Entity-Relationship Identification:
- Identify explicit/nascent entities (Subject: <entity>, Measurement: <metric>, Context: <domain>)
- Map to required schema components: 
a. Core tables (1 per entity) 
b. Attribute columns (2-5 per entity) 
c. Temporal anchors (DATE-type columns)
d. Relationship connectors (FK chains)

B) Gap Detection Matrix:
1. Verify existence per entity:
   - Mandatory: Primary table with PK;
   - Operational: Columns for filtering/aggregation;
   - Temporal: Valid time dimension for range queries;
   - Relational: Connected via FK path to relevant tables.
2. Flag missing components as [UNRESOLVED: component_type].

# Phase 2: Contextual Re-alignment**
A) Semantic Binding Process:
1. Maintain original lexical anchors (technical nouns/verbs)
2. Inject missing elements using schema-adaptive phrasing:
   - Tables: "using <added>table_X</added> containing [entity] records"
   - Columns: "tracking <added>column_Y</added> for [function]"
   - Temporal: "between [start] and [end] in <added>time_column</added>"
   - Relationships: "connected through <added>FK_relationship</added>"

B) Query Intent Preservation Check:
- Ensure rewritten question requires ALL identified [UNRESOLVED] components
- Verify no new assumptions beyond original intent

# Phase 3: Precision Output Formatting**
Structured as:
"In a system tracking [core entities], how to [action verb] <added>using [missing_tables]</added>? Analyze [base_metrics] with <added>[missing_columns]</added> for [operations], filtered through [time_constraints] and correlated via <added>[missing_relationships]</added>."

# Constraints:
1. Mandatory <added> tags ONLY for schema additions
2. Preserve original question's technical vocabulary
3. Output EXCLUSIVELY the reformulated question
4. Strict JSON avoidance

[Inputs]
Question: {question}
Schemas: {context}

Rewritten Question:
"""

# Prompt For Step Two (Single-DB)
LOCATE_TEMPLATE = """### [Role]
You are a highly experienced database scientist and data analyst with deep expertise in database theory, SQL specifications, and rigorous data validation techniques. Your task is to filter out tables and columns from the given database schema that are completely irrelevant to constructing correct SQL statements derived from a natural language query.

### [Process Guidance]:
1. Verification of Relevance:
(a) Irrelevance Assurance: Confirm that each field (table column) removed is 100% irrelevant to the natural language question and does not affect the final SQL statement.
(b) Prevent False Negatives: Adopt a conservative approach that always prioritizes retaining any elements that might have an implicit connection to the query context.

2.Rigorous Analysis Process:
(a) Context Binding:
# Identify 2-5 primary entities (with any potential aliases or variants).
# Detect hidden constraints such as temporal windows, spatial boundaries, and aggregation needs from the query.
(b) Adaptive Filtering:
# Conservative Exclusion:
- Remove a field only if there is complete lexical isolation (no partial or semantic match with the query tokens, considering case-insensitive comparisons and domain-specific ontologies).
- Exclude a table or column if it lacks any join path (within 3 hops) to the identified primary entities or relevant operational context.
- Discard fields where there is a definitive type contradiction or proven cardinality mismatch.
# Relevance Tagging:
- Instead of immediate removal, flag fields as potential low-usage candidates, legacy context elements, or system artifacts if they require further review.

3. Mandatory Retention Rules:
(a) Retain all primary key (PK) and foreign key (FK) fields, including their composite counterparts and any column that participates in multiple join conditions.
(b) Ensure that all numeric, temporal, or geometric fields—especially those related to the query’s implicit domains—are maintained.

4. Final Validation Steps:
(a) Verify that every core entity table still maintains its essential join paths.
(b) Confirm that no critical numeric or temporal fields have been inadvertently removed.
(b) Achieve an efficient yet accurate reduction of schema noise (targeting around 40-70% reduction) without compromising SQL-critical elements or introducing errors.

### [Output Requirements]
After performing the schema pruning and validation steps, compile your final decision into a single Python list. This list must contain all the irrelevant fields agreed upon, where each entry is formatted as [<table>.<column>] (e.g., ['users.age', 'orders.discount_code', 'products.supplier_id']). Do not include any additional text or commentary in the output.

### [Process Begin]
[Question]
{question}.

[Provided Database Schema]
{context}.

### Only output a single Python list.
"""

# Prompt For Step Two (Multi-DB)
MULTI_LOCATE_TEMPLATE = """
You are a database expert, who has professional knowledge of databases and highly proficient in writing SQL statements.
On the basis of comprehensive understanding the natural language problem, 
let's think step by step to determine the only one database which has the most sufficient data tables and data fields to construct the exact SQL statements.
#
Strictly Output a unique database name without any irrelevant content.
### 
Here are a few reference examples that may help you complete this task. 
#
Database Table Creation Statements:
### 
Database Name: student_transcripts_tracking

CREATE TABLE `Degree_Programs` (
`degree_program_id` INTEGER PRIMARY KEY COMMENT 'Unique identifier for the degree program',
`department_id` INTEGER NOT NULL COMMENT 'Identifier for the associated department',
`degree_summary_name` VARCHAR(255) COMMENT 'Summary name of the degree program',
`degree_summary_description` VARCHAR(255) COMMENT 'Description of the degree program',
`other_details` VARCHAR(255) COMMENT 'Other details about the degree program',
FOREIGN KEY (`department_id` ) REFERENCES `Departments`(`department_id` )
);

CREATE TABLE `Semesters` (
`semester_id` INTEGER PRIMARY KEY COMMENT 'Unique identifier for the semester',
`semester_name` VARCHAR(255) COMMENT 'Name of the semester',
`semester_description` VARCHAR(255) COMMENT 'Description of the semester',
`other_details` VARCHAR(255) COMMENT 'Other details about the semester'
);

CREATE TABLE `Students` (
`student_id` INTEGER PRIMARY KEY COMMENT 'Unique identifier for the student',
`current_address_id` INTEGER NOT NULL COMMENT 'Identifier for the current address',
`permanent_address_id` INTEGER NOT NULL COMMENT 'Identifier for the permanent address',
`first_name` VARCHAR(80) COMMENT 'Student''s first name',
`middle_name` VARCHAR(40) COMMENT 'Student''s middle name',
`last_name` VARCHAR(40) COMMENT 'Student''s last name',
`cell_mobile_number` VARCHAR(40) COMMENT 'Student''s mobile number',
`email_address` VARCHAR(40) COMMENT 'Student''s email address',
`ssn` VARCHAR(40) COMMENT 'Student''s social security number',
`date_first_registered` DATETIME COMMENT 'Date the student first registered',
`date_left` DATETIME COMMENT 'Date the student left',
`other_student_details` VARCHAR(255) COMMENT 'Other details about the student',
FOREIGN KEY (`current_address_id` ) REFERENCES `Addresses`(`address_id` ),
FOREIGN KEY (`permanent_address_id` ) REFERENCES `Addresses`(`address_id` )
);

CREATE TABLE `Student_Enrolment` (
`student_enrolment_id` INTEGER PRIMARY KEY COMMENT 'Unique identifier for student enrolment',
`degree_program_id` INTEGER NOT NULL COMMENT 'Identifier for the associated degree program',
`semester_id` INTEGER NOT NULL COMMENT 'Identifier for the associated semester',
`student_id` INTEGER NOT NULL COMMENT 'Identifier for the associated student',
`other_details` VARCHAR(255) COMMENT 'Other details about the enrolment',
FOREIGN KEY (`degree_program_id` ) REFERENCES `Degree_Programs`(`degree_program_id` ),
FOREIGN KEY (`semester_id` ) REFERENCES `Semesters`(`semester_id` ),
FOREIGN KEY (`student_id` ) REFERENCES `Students`(`student_id` )
);

CREATE TABLE `Student_Enrolment_Courses` (
`student_course_id` INTEGER PRIMARY KEY COMMENT 'Unique identifier for student course enrolment',
`course_id` INTEGER NOT NULL COMMENT 'Identifier for the associated course',
`student_enrolment_id` INTEGER NOT NULL COMMENT 'Identifier for the associated student enrolment',
FOREIGN KEY (`course_id` ) REFERENCES `Courses`(`course_id` ),
FOREIGN KEY (`student_enrolment_id` ) REFERENCES `Student_Enrolment`(`student_enrolment_id` )
);

### 
Database Name: csu_1

CREATE TABLE "Campuses" (
	"Id" INTEGER PRIMARY KEY, -- Unique identifier for each campus
	"Campus" TEXT, -- Name of the campus
	"Location" TEXT, -- Geographical location of the campus
	"County" TEXT, -- County where the campus is located
	"Year" INTEGER -- Year of establishment or relevant year
);

CREATE TABLE "csu_fees" ( 
	"Campus" INTEGER PRIMARY KEY, -- Reference to campus ID
	"Year" INTEGER, -- Academic year
	"CampusFee" INTEGER, -- Fee amount for the campus
	FOREIGN KEY (Campus) REFERENCES Campuses(Id)
);

CREATE TABLE "degrees" ( 
	"Year" INTEGER, -- Academic year
	"Campus" INTEGER, -- Reference to campus ID
	"Degrees" INTEGER, -- Number of degrees awarded
	PRIMARY KEY (Year, Campus),
	FOREIGN KEY (Campus) REFERENCES Campuses(Id)
);

CREATE TABLE "discipline_enrollments" ( 
	"Campus" INTEGER, -- Reference to campus ID
	"Discipline" INTEGER, -- Discipline or field of study ID
	"Year" INTEGER, -- Academic year
	"Undergraduate" INTEGER, -- Number of undergraduate students
	"Graduate" INTEGER, -- Number of graduate students
	PRIMARY KEY (Campus, Discipline),
	FOREIGN KEY (Campus) REFERENCES Campuses(Id)
);

CREATE TABLE "enrollments" ( 
	"Campus" INTEGER, -- Reference to campus ID
	"Year" INTEGER, -- Academic year
	"TotalEnrollment_AY" INTEGER, -- Total enrollment for the academic year
	"FTE_AY" INTEGER, -- Full-time equivalent enrollment for the academic year
	PRIMARY KEY(Campus, Year),
	FOREIGN KEY (Campus) REFERENCES Campuses(Id)
);

CREATE TABLE "faculty" ( 
	"Campus" INTEGER, -- Reference to campus ID
	"Year" INTEGER, -- Academic year
	"Faculty" REAL, -- Number of faculty members
	FOREIGN KEY (Campus) REFERENCES Campuses(Id) 
);
#
Question: Find the semester when both Master students and Bachelor students got enrolled in.
Analysis: Let's think step by step to determine the exact database corresponding to the question by using the provided database schemas.
Step 1: Key Requirements of the Question.We are tasked with finding the semester when both Master students and Bachelor students got enrolled.This implies that we need data about students' enrollments (Bachelor and Master), and semester information.
Step 2: Database Schemas Comparison.The student_transcripts_tracking schema directly links students to their degree programs and specific semesters. This schema allows us to distinguish between Bachelor's and Master's students, and the presence of the Semesters table enables precise filtering of enrollment data by semester, making it ideal for identifying when both groups were enrolled.The csu_1 schema lacks the necessary granularity for this query. It does not provide detailed semester data or a way to directly link students to their program type (Bachelor's or Master's). While it offers some aggregate enrollment information, it does not enable filtering based on specific semesters, making it insufficient for answering the query about when both Bachelor’s and Master’s students were enrolled together in a semester.
Step 3: Conclusion.The gold database is student_transcripts_tracking because it contains the required detailed data to answer the query about semester enrollments for both Master and Bachelor students.
Database Name: student_transcripts_tracking
### 
The reason work for this round officially begin now.
#
Relevant database Table Creation Statements:
{context}
#
Question:{question}
Output Database Name:
"""

# Prompt For Step Three
SCHEMA_LINKING_MANUAL_TEMPLATE = """
You are a database expert who is highly proficient in writing SQL statements. 
For a natural language question , you job is to identify and extract the correct data tables and data fields from database creation statements,
which is strictly necessary for the accurate SQL statement corresponding to the question. 
#
Strictly output the results in a python list format:
[<data table name>.<data field name>=<Analysis tokens based on the Question>...]
e.g. "movies" and "ratings" are two datatable in one database,then one possible output as following:
[movies.movie_release_year="Which year", movies.movie_title="title of the movie", ratings.rating_score="rating score"]
#
{few_examples}
# The extraction work for this round officially begin now.
Database Table Creation Statements:
{context_str}
#
Question: {question}
Answer:
"""

SCHEMA_LINKING_FEW_EXAMPLES = """
### 
Here are a few reference examples that may help you complete this task. 
### 
Database Table Creation Statements:
#
Following is the whole table creation statement for the database popular_movies
CREATE TABLE movies (
        movie_id INTEGER NOT NULL, 
        movie_title TEXT, 
        movie_release_year INTEGER, 
        movie_url TEXT, 
        movie_title_language TEXT, 
        movie_popularity INTEGER, 
        movie_image_url TEXT, 
        director_id TEXT, 
        director_name TEXT, 
        director_url TEXT, 
        PRIMARY KEY (movie_id)
)
CREATE TABLE ratings (
        movie_id INTEGER, 
        rating_id INTEGER, 
        rating_url TEXT, 
        rating_score INTEGER, 
        rating_timestamp_utc TEXT, 
        critic TEXT, 
        critic_likes INTEGER, 
        critic_comments INTEGER, 
        user_id INTEGER, 
        user_trialist INTEGER, 
        user_subscriber INTEGER, 
        user_eligible_for_trial INTEGER, 
        user_has_payment_method INTEGER, 
        FOREIGN KEY(movie_id) REFERENCES movies (movie_id), 
        FOREIGN KEY(user_id) REFERENCES lists_users (user_id), 
        FOREIGN KEY(rating_id) REFERENCES ratings (rating_id), 
        FOREIGN KEY(user_id) REFERENCES ratings_users (user_id)
)
Question: Which year has the least number of movies that was released and what is the title of the movie in that year that has the highest number of rating score of 1?
Hint: least number of movies refers to MIN(movie_release_year); highest rating score refers to MAX(SUM(movie_id) where rating_score = '1')
Analysis: Let’s think step by step. In the question , we are asked:
"Which year" so we need column = [movies.movie_release_year]
"number of movies" so we need column = [movies.movie_id]
"title of the movie" so we need column = [movies.movie_title]
"rating score" so we need column = [ratings.rating_score]
Hint also refers to the columns = [movies.movie_release_year, movies.movie_id, ratings.rating_score]
Based on the columns and tables, we need these Foreign_keys = [movies.movie_id = ratings.movie_id].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [1]. So the Schema_links are:
Answer: [movies.movie_release_year="Which year", movies.movie_title="title of the movie", ratings.rating_score="rating score", movies.movie_id="number of movies","Foreign_keys": [movies.movie_id = ratings.movie_id]]


#
Following is the whole table creation statement for the database user_list
CREATE TABLE lists (
        user_id INTEGER, 
        list_id INTEGER NOT NULL, 
        list_title TEXT, 
        list_movie_number INTEGER, 
        list_update_timestamp_utc TEXT, 
        list_creation_timestamp_utc TEXT, 
        list_followers INTEGER, 
        list_url TEXT, 
        list_comments INTEGER, 
        list_description TEXT, 
        list_cover_image_url TEXT, 
        list_first_image_url TEXT, 
        list_second_image_url TEXT, 
        list_third_image_url TEXT, 
        PRIMARY KEY (list_id), 
        FOREIGN KEY(user_id) REFERENCES lists_users (user_id)
)
CREATE TABLE lists_users (
        user_id INTEGER NOT NULL, 
        list_id INTEGER NOT NULL, 
        list_update_date_utc TEXT, 
        list_creation_date_utc TEXT, 
        user_trialist INTEGER, 
        user_subscriber INTEGER, 
        user_avatar_image_url TEXT, 
        user_cover_image_url TEXT, 
        user_eligible_for_trial TEXT, 
        user_has_payment_method TEXT, 
        PRIMARY KEY (user_id, list_id), 
        FOREIGN KEY(list_id) REFERENCES lists (list_id), 
        FOREIGN KEY(user_id) REFERENCES lists (user_id)
)
Question: Among the lists created by user 4208563, which one has the highest number of followers? Indicate how many followers it has and whether the user was a subscriber or not when he created the list.
Hint: User 4208563 refers to user_id;highest number of followers refers to MAX(list_followers); user_subscriber = 1 means that the user was a subscriber when he created the list; user_subscriber = 0 means the user was not a subscriber when he created the list (to replace)
Analysis: Let’s think step by step. In the question , we are asked:
"user" so we need column = [lists_users.user_id]
"number of followers" so we need column = [lists.list_followers]
"user was a subscriber or not" so we need column = [lists_users.user_subscriber]
Hint also refers to the columns = [lists_users.user_id,lists.list_followers,lists_users.user_subscriber]
Based on the columns and tables, we need these Foreign_keys = [lists.user_id = lists_user.user_id,lists.list_id = lists_user.list_id].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [1, 4208563]. So the Schema_links are:
Answer: [lists.list_followers= "number of followers",lists_users.user_subscriber="user was a subscriber or not",lists_user.user_id= "user","Foreign_keys": [lists.user_id = lists_user.user_id,lists.list_id = lists_user.list_id]]

###
"""

# Prompt For Basic RAG
SCHEMA_LINKING_TEMPLATE = """
You are a database expert who is highly proficient in writing SQL statements. 
For a natural language question , you job is to identify and extract the correct data tables and data fields from database creation statements,
which is necessary for constructing the accurate SQL statement corresponding to the question. 
#
Strictly ensure the output is a Python list object:
[<data table name>.<data field name>=<Analysis tokens based on the Question>...]
e.g. "movies" and "ratings" are two datatable in one database,then one possible output as following:
[movies.movie_release_year="Which year", movies.movie_title="title of the movie", ratings.rating_score="rating score"]
#
{few_examples}
# The extraction work for this round officially begin now.
Database Table Creation Statements:
{{context_str}}
#
Question: {question}
Answer:
"""
DEFAULT_PROMPT_TEMPLATE = SCHEMA_LINKING_TEMPLATE
