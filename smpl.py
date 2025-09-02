"-----------------------------------------------------------------------------------"
"""                    BOOLEAN CONJUNCTIVE QUERY (WITH BRACKETS)                  """
"-----------------------------------------------------------------------------------"
import re
import collections


# EXTRACTING DOCUMENTS FROM TEXT :
def extract_documents_from_string(text):
    documents = {}
    parts = re.split(r'(Document\s+\d+)', text)
    current_doc_num = None
    current_doc_content = []

    for part in parts:
        if part.startswith('Document'):
            if current_doc_num is not None and current_doc_content:
                documents[current_doc_num] = "".join(current_doc_content).strip()
                current_doc_content = []
            match = re.search(r'Document\s+(\d+)', part)
            if match:
                current_doc_num = int(match.group(1))
        elif current_doc_num is not None:
            lines = part.splitlines()
            content_lines = [line for line in lines if "********************************************" not in line]
            current_doc_content.extend(content_lines)

    if current_doc_num is not None and current_doc_content:
         documents[current_doc_num] = "".join(current_doc_content).strip()
    return documents


# READING FILE :
with open("documents.txt", 'r', encoding='utf-8') as f:
    file_content_from_drive = f.read()
extracted_docs = extract_documents_from_string(file_content_from_drive)
for i, (doc_num, content) in enumerate(extracted_docs.items()): #printing only 5 docs
    if i >= 5:
        break
    print(f"--- Document {doc_num} ---")
    print(content)
    print("-" * 20)


# CREATING INVERTED INDEX AND PROCESS SIMPLE BOOLEAN QUERIES :
def create_inverted_index(documents):
    inverted_index = collections.defaultdict(list)
    for doc_num, content in documents.items():
        words = re.findall(r'\b\w+\b', content.lower())
        for word in words:
            if doc_num not in inverted_index[word]:
                inverted_index[word].append(doc_num)
    return inverted_index

def remove_stop_words(inverted_index, num_stop_words=10):
    word_frequencies = sorted(inverted_index.items(), key=lambda item: len(item[1]), reverse=True)
    stop_words = [word for word, _ in word_frequencies[:num_stop_words]]
    new_index = {word: postings for word, postings in inverted_index.items() if word not in stop_words}
    return new_index, stop_words

def tokenize_query(query):
    # Tokenize: words, AND, OR, NOT, and parentheses
    return re.findall(r'\b\w+\b|AND|OR|NOT|\(|\)', query.upper())

def eval_boolean_query(tokens, inverted_index, all_docs):
    def eval_expr(tokens):
        def parse_operand():
            token = tokens.pop(0)
            if token == '(':
                result = parse_or()
                tokens.pop(0)
                return result
            elif token == 'NOT':
                operand = parse_operand()
                return all_docs - operand
            else:
                return set(inverted_index.get(token.lower(), []))

        def parse_and():
            left = parse_operand()
            while tokens and tokens[0] == 'AND':
                tokens.pop(0)
                right = parse_operand()
                left = left & right
            return left

        def parse_or():
            left = parse_and()
            while tokens and tokens[0] == 'OR':
                tokens.pop(0)
                right = parse_and()
                left = left | right
            return left

        return parse_or()

    return sorted(list(eval_expr(tokens)))


# MAIN FUNCTION :
inverted_index = create_inverted_index(extracted_docs)
inverted_index_no_stopwords, stop_words = remove_stop_words(inverted_index)

all_doc_ids = set(extracted_docs.keys())

initial_index_size = len(inverted_index)
index_size_no_stopwords = len(inverted_index_no_stopwords)
print(f"Initial index size (unique terms): {initial_index_size}")
print(f"Index size after removing stop words: {index_size_no_stopwords}")
print("-" * 50)

# Example Queries
queries = [
    "london AND ltd",
    "training AND services",
    "council OR service",
    "training AND NOT services",
    "(council OR training) AND NOT service",
    "library and training or services and council"
]

for query in queries:
    tokens = tokenize_query(query)
    results = eval_boolean_query(tokens, inverted_index_no_stopwords, all_doc_ids)
    print(f"Results for query '{query}': {results}")
    print("-" * 50)


# CNF IMPLEMENTATION :
def evaluate_cnf_query(inverted_index, cnf_query):
    clauses = [clause.strip() for clause in cnf_query.lower().split(' and ')]
    doc_sets = []

    for clause in clauses:
        # Split terms by OR
        terms = [term.strip() for term in clause.split(' or ')]
        postings_lists = [set(inverted_index.get(term, [])) for term in terms]
        if not postings_lists:
            continue
        # OR within a clause
        clause_docs = set.union(*postings_lists)
        doc_sets.append(clause_docs)
    if not doc_sets:
        return []
    result_docs = set.intersection(*doc_sets)
    return sorted(list(result_docs))


# DNF IMPLEMENTATION :
def evaluate_dnf_query(inverted_index, dnf_query):
    # Split terms by OR to get individual conjunctions
    and_groups = [group.strip() for group in dnf_query.lower().split(' or ')]
    result_docs = set()

    for group in and_groups:
        terms = [term.strip() for term in group.split(' and ')]
        postings_lists = [set(inverted_index.get(term, [])) for term in terms]
        if not postings_lists:
            continue
        # AND within a group
        group_docs = set.intersection(*postings_lists)
        result_docs.update(group_docs)

    return sorted(list(result_docs))


# CONJUCTIVE QUERIES AND MIXED QUERIES :
cnf_query = "library or training and services or council"
dnf_query = "library and training or services and council"

cnf_results = evaluate_cnf_query(inverted_index_no_stopwords, cnf_query)
dnf_results = evaluate_dnf_query(inverted_index_no_stopwords, dnf_query)

print(f"Results for CNF query '{cnf_query}': {cnf_results}")
print(f"Results for DNF query '{dnf_query}': {dnf_results}")

"-----------------------------------------------------------------------------------"
"                                     END                                           "
"-----------------------------------------------------------------------------------"









"-----------------------------------------------------------------------------------"
"""                               INVERTED INDEX                                  """
"-----------------------------------------------------------------------------------"

class InvertedIndex:
    def __init__(self, docs):
        self.docs = docs
        self.index = {}
        self._build()
    def _build(self):
        """Build inverted index"""
        for i, doc in enumerate(self.docs):
            for term in set(doc.lower().split()):
                if term not in self.index:
                    self.index[term] = []
                self.index[term].append(i)
    def get(self, term):
        """Get posting list"""
        return self.index.get(term.lower(), [])
    def AND(self, list1, list2):
        """Intersect two sorted lists"""
        return [x for x in list1 if x in list2]
    def OR(self, list1, list2):
        """Union two lists"""
        return sorted(set(list1 + list2))
    def NOT(self, posting_list):
        """Complement of posting list"""
        all_docs = list(range(len(self.docs)))
        return [x for x in all_docs if x not in posting_list]
    def optimize_terms(self, terms, operation='and'):
        """Sort terms by posting list length for optimal processing"""
        term_lengths = [(term, len(self.get(term))) for term in terms]
        if operation == 'and':
            # For AND: process shortest lists first (fewer intersections)
            return [term for term, _ in sorted(term_lengths, key=lambda x: x[1])]
        else:  # OR
            # For OR: process longest lists first (build result faster)
            return [term for term, _ in sorted(term_lengths, key=lambda x: x[1], reverse=True)]
    def search(self, query):
        """Optimized boolean search"""
        q = query.lower()
        if ' and ' in q:
            terms = [t.strip() for t in q.split(' and ')]
            # Optimize: shortest posting lists first
            terms = self.optimize_terms(terms, 'and')
            result = self.get(terms[0])
            for term in terms[1:]:
                result = self.AND(result, self.get(term))
                if not result:  # Early termination
                    break
            return result
        elif ' or ' in q:
            terms = [t.strip() for t in q.split(' or ')]
            # Optimize: longest posting lists first
            terms = self.optimize_terms(terms, 'or')
            result = self.get(terms[0])
            for term in terms[1:]:
                result = self.OR(result, self.get(term))
            return result
        elif ' not ' in q:
            pos, neg = q.split(' not ')
            pos_list = self.get(pos.strip())
            neg_list = self.get(neg.strip())
            return self.AND(pos_list, self.NOT(neg_list))
        else:
            return self.get(q)
# Usage with optimization demo
docs = ["cat dog bird", "dog bird", "cat mouse", "bird eagle", "mouse cat"]
idx = InvertedIndex(docs)
print("Index:", idx.index)
# Show optimization in action
print("\nQuery: 'cat and bird and dog'")
terms = ['cat', 'bird', 'dog']
print("Posting list sizes:")
for term in terms:
    print(f"  {term}: {len(idx.get(term))} docs")
optimized = idx.optimize_terms(terms, 'and')
print(f"Optimized order: {optimized}")  # Shortest first
print(f"Result: {idx.search('cat and bird and dog')}")
print(f"\nOR optimization:")
or_optimized = idx.optimize_terms(terms, 'or')
print(f"OR order: {or_optimized}")  # Longest first

"-----------------------------------------------------------------------------------"
"                                     END                                           "
"-----------------------------------------------------------------------------------"








"-----------------------------------------------------------------------------------"
"""                         TERM DOCUMENT BOOLEAN MODEL                           """
"-----------------------------------------------------------------------------------"

class TermDocumentBooleanModel:
    def __init__(self, documents):
        self.documents = documents
        self.vocab = []
        self.term_doc_matrix = []
        self._build_matrix()
    
    def _build_matrix(self):
        """Build term-document matrix"""
        # Get all unique terms
        all_terms = set()
        processed_docs = []
        
        for doc in self.documents:
            tokens = doc.lower().split()
            processed_docs.append(tokens)
            all_terms.update(tokens)
        
        self.vocab = sorted(all_terms)
        
        # Build matrix: rows=terms, cols=documents
        self.term_doc_matrix = []
        for term in self.vocab:
            row = []
            for doc_tokens in processed_docs:
                row.append(1 if term in doc_tokens else 0)
            self.term_doc_matrix.append(row)
    
    def get_term_vector(self, term):
        """Get document vector for a term"""
        term = term.lower()
        if term not in self.vocab:
            return [0] * len(self.documents)
        
        term_idx = self.vocab.index(term)
        return self.term_doc_matrix[term_idx]
    
    def boolean_and(self, term1, term2):
        """Boolean AND operation"""
        vec1 = self.get_term_vector(term1)
        vec2 = self.get_term_vector(term2)
        return [a & b for a, b in zip(vec1, vec2)]
    
    def boolean_or(self, term1, term2):
        """Boolean OR operation"""
        vec1 = self.get_term_vector(term1)
        vec2 = self.get_term_vector(term2)
        return [a | b for a, b in zip(vec1, vec2)]
    
    def boolean_not(self, term):
        """Boolean NOT operation"""
        vec = self.get_term_vector(term)
        return [1 - x for x in vec]
    
    def search(self, query):
        """Search with boolean operators"""
        query = query.lower().strip()
        
        # Single term
        if ' ' not in query:
            result_vector = self.get_term_vector(query)
        
        # AND operation
        elif ' and ' in query:
            terms = [t.strip() for t in query.split(' and ')]
            result_vector = self.get_term_vector(terms[0])
            for term in terms[1:]:
                term_vec = self.get_term_vector(term)
                result_vector = [a & b for a, b in zip(result_vector, term_vec)]
        
        # OR operation
        elif ' or ' in query:
            terms = [t.strip() for t in query.split(' or ')]
            result_vector = self.get_term_vector(terms[0])
            for term in terms[1:]:
                term_vec = self.get_term_vector(term)
                result_vector = [a | b for a, b in zip(result_vector, term_vec)]
        
        # NOT operation
        elif ' not ' in query:
            parts = query.split(' not ')
            pos_term = parts[0].strip()
            neg_term = parts[1].strip()
            
            pos_vec = self.get_term_vector(pos_term)
            neg_vec = self.get_term_vector(neg_term)
            neg_vec = [1 - x for x in neg_vec]  # NOT operation
            result_vector = [a & b for a, b in zip(pos_vec, neg_vec)]
        
        else:
            result_vector = [0] * len(self.documents)
        
        # Return document IDs where result is 1
        return [i for i, val in enumerate(result_vector) if val == 1]
    
    def print_matrix(self):
        """Print term-document matrix"""
        print("Term-Document Matrix:")
        print("Terms\\Docs", end="")
        for i in range(len(self.documents)):
            print(f"\tD{i}", end="")
        print()
        
        for i, term in enumerate(self.vocab):
            print(f"{term:<10}", end="")
            for val in self.term_doc_matrix[i]:
                print(f"\t{val}", end="")
            print()

# Usage Example
if __name__ == "__main__":
    # Sample documents
    docs = [
        "information retrieval system",
        "database search query",
        "information system database",
        "web search engine",
        "query processing system"
    ]
    
    model = TermDocumentBooleanModel(docs)
    model.print_matrix()
    
    print("\nSearch Results:")
    print("'information':", model.search("information"))
    print("'information and system':", model.search("information and system"))
    print("'search or query':", model.search("search or query"))
    print("'system not database':", model.search("system not database"))

"-----------------------------------------------------------------------------------"
"                                     END                                           "
"-----------------------------------------------------------------------------------"






"-----------------------------------------------------------------------------------"
"""                            VECTOR SPACE MODEL                                """
"-----------------------------------------------------------------------------------"

import math
from collections import Counter

class VectorSpaceModel:
    def __init__(self, docs):
        self.docs = docs
        self.vocab, self.tf_matrix = self._preprocess()
        self.idf = self._compute_idf()
        self.tfidf_matrix = self._compute_tfidf()
    
    def _preprocess(self):
        """Integrated preprocessing"""
        processed = []
        vocab_set = set()
        
        for doc in self.docs:
            tokens = doc.lower().split()
            processed.append(tokens)
            vocab_set.update(tokens)
        
        vocab = sorted(vocab_set)
        
        # Build TF matrix
        tf_matrix = []
        for doc_tokens in processed:
            counts = Counter(doc_tokens)
            tf_row = [counts.get(term, 0) for term in vocab]
            tf_matrix.append(tf_row)
        
        print("\n TF-Matrix :\n", tf_matrix, "\n")
        return vocab, tf_matrix
    
    def _compute_idf(self):
        """IDF calculation"""
        N = len(self.tf_matrix)
        idf = []
        for term_idx in range(len(self.vocab)):
            df = sum(1 for doc in self.tf_matrix if doc[term_idx] > 0)
            idf.append(math.log(N / df) if df > 0 else 0)
        return idf
    
    def _compute_tfidf(self):
        """TF-IDF matrix"""
        return [[tf * self.idf[i] for i, tf in enumerate(doc)] 
                for doc in self.tf_matrix]
    
    def _query_to_vector(self, query):
        """Query to TF-IDF vector"""
        query_tf = Counter(query.lower().split())
        return [query_tf.get(term, 0) * self.idf[i] 
                for i, term in enumerate(self.vocab)]
    
    def cosine_similarity(self, v1, v2):
        """Cosine similarity"""
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(a * a for a in v2))
        return dot / (mag1 * mag2) if mag1 and mag2 else 0
    
    def jaccard_coefficient(self, v1, v2):
        """Jaccard coefficient for binary vectors"""
        # Convert to binary
        b1 = [1 if x > 0 else 0 for x in v1]
        b2 = [1 if x > 0 else 0 for x in v2]
        
        intersection = sum(a & b for a, b in zip(b1, b2))
        union = sum(a | b for a, b in zip(b1, b2))
        
        return intersection / union if union > 0 else 0
    
    def dice_coefficient(self, v1, v2):
        """Dice coefficient"""
        b1 = [1 if x > 0 else 0 for x in v1]
        b2 = [1 if x > 0 else 0 for x in v2]
        
        intersection = sum(a & b for a, b in zip(b1, b2))
        total = sum(b1) + sum(b2)
        
        return (2 * intersection) / total if total > 0 else 0
    
    def dot_product(self, v1, v2):
        """Simple dot product"""
        return sum(a * b for a, b in zip(v1, v2))
    
    def search(self, query, similarity='cosine', top_k=5):
        """Search with different similarity measures"""
        qvec = self._query_to_vector(query)
        
        similarities = []
        for doc_id, dvec in enumerate(self.tfidf_matrix):
            if similarity == 'cosine':
                sim = self.cosine_similarity(qvec, dvec)
            elif similarity == 'jaccard':
                sim = self.jaccard_coefficient(qvec, dvec)
            elif similarity == 'dice':
                sim = self.dice_coefficient(qvec, dvec)
            elif similarity == 'dot':
                sim = self.dot_product(qvec, dvec)
            else:
                sim = self.cosine_similarity(qvec, dvec)
            
            similarities.append((doc_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Usage
docs = ["information retrieval system", "machine learning data", "web search engine"]
vsm = VectorSpaceModel(docs)

# Different similarity measures
print("Cosine:", vsm.search("information system", 'cosine'))
print("Jaccard:", vsm.search("information system", 'jaccard'))
print("Dice:", vsm.search("information system", 'dice'))
print("Dot Product:", vsm.search("information system", 'dot'))

"-----------------------------------------------------------------------------------"
"                                     END                                           "
"-----------------------------------------------------------------------------------"





"-----------------------------------------------------------------------------------"
"""                                 BIM (BINARY INDEPENDENCE MODEL)                 """
"-----------------------------------------------------------------------------------"

import math

class BinaryIndependenceModel:
    def __init__(self, docs):
        self.docs = docs
        self.vocab, self.binary_matrix = self._preprocess()
        self.N_d = len(self.binary_matrix)  # Total documents
    
    def _preprocess(self):
        """Build vocabulary and binary matrix"""
        processed = [set(doc.lower().split()) for doc in self.docs]
        vocab = sorted({t for doc in processed for t in doc})
        binary_matrix = [[int(term in doc) for term in vocab] for doc in processed]
        return vocab, binary_matrix
    
    def _get_dk(self, term_idx):
        return sum(doc[term_idx] for doc in self.binary_matrix)
    
    def _estimate(self, query_terms, relevant_docs=None):
        """Unified estimation: Phase I (no relevance) or Phase II (with relevance)"""
        estimates = {}
        N_r = len(relevant_docs) if relevant_docs else 0
        
        for term in query_terms:
            if term not in self.vocab: 
                continue
            term_idx = self.vocab.index(term)
            d_k = self._get_dk(term_idx)
            
            if not relevant_docs:  # Phase I
                p_k, q_k = 0.5, (d_k + 0.5) / (self.N_d + 1)
                estimates[term] = {"d_k": d_k, "p_k": p_k, "q_k": q_k}
            else:  # Phase II
                r_k = sum(self.binary_matrix[doc_id][term_idx] for doc_id in relevant_docs)
                p_k = (r_k + 0.5) / (N_r + 1)
                q_k = (d_k - r_k + 0.5) / (self.N_d - N_r + 1)
                estimates[term] = {"r_k": r_k, "d_k": d_k, "N_r": N_r, "p_k": p_k, "q_k": q_k}
        
        return estimates
    
    def calculate_rsv(self, doc_id, query_terms, estimates):
        rsv = 0
        for term in query_terms:
            if term not in estimates: 
                continue
            term_idx = self.vocab.index(term)
            p_k, q_k = estimates[term]["p_k"], estimates[term]["q_k"]
            
            if self.binary_matrix[doc_id][term_idx]:  # term in doc
                if p_k > 0 and q_k > 0:
                    rsv += math.log(p_k / q_k)
            else:  # term not in doc
                if p_k < 1 and q_k < 1:
                    rsv += math.log((1 - p_k) / (1 - q_k))
        return rsv
    
    def _search(self, query, estimates, top_k):
        query_terms = query.lower().split()
        scores = [(doc_id, self.calculate_rsv(doc_id, query_terms, estimates)) for doc_id in range(self.N_d)]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    def search_phase1(self, query, top_k=5):
        return self._search(query, self._estimate(query.lower().split()), top_k)
    
    def search_phase2(self, query, relevant_docs, top_k=5):
        return self._search(query, self._estimate(query.lower().split(), relevant_docs), top_k)
    
    def print_estimates(self, query_terms, relevant_docs=None):
        estimates = self._estimate(query_terms, relevant_docs)
        if relevant_docs is None:
            print("=== PHASE I ESTIMATES ===")
            for t, e in estimates.items():
                print(f"{t}: d_k={e['d_k']}, p_k={e['p_k']:.3f}, q_k={e['q_k']:.3f}")
        else:
            print("=== PHASE II ESTIMATES ===")
            for t, e in estimates.items():
                print(f"{t}: r_k={e['r_k']}, d_k={e['d_k']}, N_r={e['N_r']}, p_k={e['p_k']:.3f}, q_k={e['q_k']:.3f}")

# Usage Example
if __name__ == "__main__":
    docs = [
        "information retrieval system",
        "database search query", 
        "information system database",
        "web search engine",
        "query processing system"
    ]
    
    bim = BinaryIndependenceModel(docs)
    
    query = "information system"
    query_terms = query.split()
    
    print("=== PHASE I (No Relevance Info) ===")
    bim.print_estimates(query_terms)
    results1 = bim.search_phase1(query)
    print(f"Phase I Results: {results1}")
    
    print("\n=== PHASE II (With Relevance Feedback) ===")
    relevant_docs = [0, 2]  # Assume docs 0,2 are relevant
    bim.print_estimates(query_terms, relevant_docs)
    results2 = bim.search_phase2(query, relevant_docs)
    print(f"Phase II Results: {results2}")

"-----------------------------------------------------------------------------------"
"                                     END                                           "
"-----------------------------------------------------------------------------------"
