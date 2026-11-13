# Full Retrieved Chunk Text Audit Report

**Audit Date:** 2026-11-13  
**Execution Mode:** Strictly Read-Only (Pinecone REST Vector Fetch; zero embedding or LLM API calls; no source code or configuration changes).

---

## 1. Executive Summary

This report delivers a comprehensive, text-level relevance audit of the live RAG retrieval results across all five evaluation questions for both the **Original Top-3** (Pinecone similarity ranking) and the **Diversified Top-3** (candidate adjacency deduplication).

### Core Findings

1. **Failure of Coordinate Matching as a Relevance Proxy:**
   Previous evaluation reports (`retrieval_evaluation_report.md` and `content_aware_retrieval_evaluation_report.md`) used exact page/chunk coordinate matching against nominal gold labels (`1/1`, `2/1`, `8/3`, `16/0`, `15/1`). That proxy failed to capture actual relevance:
   - For **Q1 (DNS)**, the top-ranked chunk (`1/2`) contains the actual step-by-step DNS translation process and overlaps with the gold passage, but was scored 0 due to coordinate mismatch (`1/2` vs `1/1`).
   - For **Q3 (Serverless APIs)**, the retrieved chunks (`10/0`, `10/1`, `9/2`) contain substantive explanations of serverless architecture and operational contrasts with serverful models, but were scored 0 because nominal label `8/3` was not retrieved.
   - For **Q4 (HTTP 403)**, the nominal gold coordinate `16/0` actually cuts off *before* HTTP 403, whereas retrieved chunk `16/1` (Rank 3 in Original, Rank 2 in Diversified) contains the exact, authoritative definition (`403 Forbidden → You don’t have permission to access this resource`). Coordinate matching scored this as 0.

2. **Grounded Text-Level Performance:**
   When evaluated on actual text content and verified answer evidence:
   - **Recall@3 reaches 1.000 (100%)** across all five questions for both Original and Diversified retrieval (every question has at least one substantively sufficient answer passage in its Top-3).
   - Under coordinate matching, Recall@3 was severely underreported at **0.400**.

3. **Impact of Retrieval Diversification:**
   - **Mean Reciprocal Rank (MRR@3)** improves from **0.767** in Original Top-3 to **0.800** in Diversified Top-3.
   - For **Q4**, the diversification heuristic skips the adjacent redundant chunk `15/2`, successfully promoting the critical answer chunk `16/1` from **Rank 3 to Rank 2**, increasing Q4 Reciprocal Rank from `0.333` to `0.500`.
   - **nDCG@3** improves from **0.839** to **0.865**.
   - **Precision@3** shifts from **0.467** (Original) to **0.400** (Diversified) because diversification intentionally eliminates adjacent redundant chunks (e.g., `2/2` for Q2), replacing repeated evidence with broader document context.

---

## 2. Step 1: Identification of Required Vectors

From `backend/retrieval_evaluation_report.md` and `backend/retrieval_diversity_audit_corrected_report_v2.md`, the live retrieval candidates and gold targets are identified.

- **Target Document ID:** `ab56ef89-c787-470c-b796-7d4b9fcdb2ee`
- **Logical Document ID:** `20dcdd7c-a603-4a3f-b384-46081d625a73`
- **Target Namespace:** `revon-policies`
- **Filename:** `Pre-Read_-Basic-Understanding-on-API-1-5983.pdf`
- **Deterministic Vector ID Pattern:** `{document_id}-page-{page_number}-chunk-{chunk_index}`

### 2.1 Retrieved Top-3 Candidates

Across the five questions, 17 unique vector IDs constitute the Original and Diversified Top-3 pools:

| # | Page/Chunk | Pinecone Vector ID | Appearing In |
|---|:---:|---|---|
| 1 | `1/0` | `ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-1-chunk-0` | Q1 Diversified Rank 3 |
| 2 | `1/2` | `ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-1-chunk-2` | Q1 Original Rank 1, Diversified Rank 1 |
| 3 | `1/3` | `ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-1-chunk-3` | Q1 Original Rank 2 |
| 4 | `2/0` | `ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-2-chunk-0` | Q1 Original Rank 3, Diversified Rank 2 |
| 5 | `2/1` | `ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-2-chunk-1` | Q2 Original Rank 1, Diversified Rank 1 (Gold Q2) |
| 6 | `2/2` | `ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-2-chunk-2` | Q2 Original Rank 2 |
| 7 | `4/2` | `ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-4-chunk-2` | Q2 Diversified Rank 3 |
| 8 | `6/0` | `ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-6-chunk-0` | Q2 Original Rank 3, Diversified Rank 2 |
| 9 | `9/0` | `ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-9-chunk-0` | Q3 Original Rank 1, Diversified Rank 1 |
| 10 | `9/2` | `ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-9-chunk-2` | Q3 Diversified Rank 3 |
| 11 | `10/0` | `ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-10-chunk-0` | Q3 Original Rank 2, Diversified Rank 2 |
| 12 | `10/1` | `ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-10-chunk-1` | Q3 Original Rank 3 |
| 13 | `13/2` | `ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-13-chunk-2` | Q4 Diversified Rank 3, Q5 Diversified Rank 3 |
| 14 | `15/1` | `ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-15-chunk-1` | Q4 Orig Rank 1 / Div Rank 1, Q5 Orig Rank 1 / Div Rank 1 (Gold Q5) |
| 15 | `15/2` | `ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-15-chunk-2` | Q4 Original Rank 2, Q5 Original Rank 2 |
| 16 | `16/0` | `ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-16-chunk-0` | Q5 Original Rank 3, Diversified Rank 2 (Nominal Gold Q4) |
| 17 | `16/1` | `ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-16-chunk-1` | Q4 Original Rank 3, Diversified Rank 2 |

### 2.2 Gold Passage Reference Vectors

To perform exact text comparisons with verified gold passages, two additional vectors outside the retrieved Top-3 pools were fetched:
- `1/1` (`ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-1-chunk-1`): Verified Gold for Q1.
- `8/3` (`ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-8-chunk-3`): Verified Gold for Q3.

*(Note: Q2 gold `2/1`, Q4 nominal gold `16/0`, and Q5 gold `15/1` are already in the 17 retrieved vectors above).* Total vectors fetched: **19**.

---

## 3. Step 2: Read-Only Vector Retrieval & Metadata Validation

Vectors were retrieved directly from Pinecone using a single read-only HTTP GET request to `/vectors/fetch`:
```http
GET {PINECONE_INDEX_HOST}/vectors/fetch?ids={id1}&ids={id2}...&namespace=revon-policies
Api-Key: [REDACTED]
```

### 3.1 Metadata Validation Results

All 19 fetched records were subjected to strict validation rules:
1. `document_id == "ab56ef89-c787-470c-b796-7d4b9fcdb2ee"`: **19/19 Passed**
2. `logical_document_id == "20dcdd7c-a603-4a3f-b384-46081d625a73"`: **19/19 Passed**
3. `version == 1`: **19/19 Passed**
4. `is_active == True`: **19/19 Passed**
5. `filename == "Pre-Read_-Basic-Understanding-on-API-1-5983.pdf"`: **19/19 Passed**
6. Non-empty stored `text` field present: **19/19 Passed** (Lengths: 355 to 800 raw characters).

Zero validation errors or corrupted records were detected.

---

## 4. Step 3: Question-by-Question Gold Passage Comparison & Text Evidence

Relevance categories applied:
- **Category 2 (Complete Gold Match):** Complete verified gold passage present.
- **Category 1 (Substantive Evidence / Partial Gold):** Partial gold passage overlap or substantively sufficient answer evidence present.
- **Category 0 (Insufficient Evidence):** Lacks sufficient answer evidence to address the query.

*(Note: In the text citations below, text is presented after collapsing repetitive whitespace from the raw PDF extraction to ensure readability).*

---

### Question 1: How does DNS translate domain names to IP addresses?
- **Verified Gold Coordinate:** Page 1, Chunk 1 (`1/1`)
- **Verified Gold Passage Text (`1/1`):**
  > *"actually the power of the Internet Protocol Stack working in harmony So the very first thing your browser needs to know is where exactly this website is? It only has the name: www.codingninjas.com . But the internet doesn’t use names—it uses IP addresses . Just like when you want to call a friend, you need their number, not just their name. So, the first step is called DNS Resolution —let's see how your computer finds the IP address of the website. 1. DNS Resolution: Translating Domain Names to IP Addresses DNS (Domain Name System) is like the phonebook of the internet . Humans prefer"*

#### Retrieved Candidate Chunks (Q1):

1. **Chunk `1/2`** (Original Rank 1, Score: 0.8208; Diversified Rank 1)
   - **Vector ID:** `...-page-1-chunk-2` (790 chars raw / 598 chars normalized)
   - **Stored Text:**
     > *"Addresses DNS (Domain Name System) is like the phonebook of the internet . Humans prefer domain names ( www.google.com ), but machines need IP addresses (142.250.190.36) to route data. Why is it needed? You can’t expect users to remember the IPs of every website. DNS allows us to use easy-to-remember domain names. How it Works (Step-by-Step): 1. You type www.example.com into your browser. 2. The browser checks the local DNS cache (if you've visited recently). 3. If not found, it queries the recursive resolver (usually provided by your ISP). 4. Resolver asks the root DNS server → then the TLD"*
   - **Comparison & Evidence:**
     - Overlaps with Gold Chunk `1/1` tail (*"DNS (Domain Name System) is like the phonebook of the internet . Humans prefer"*).
     - Directly provides the technical mechanism of translation: local DNS cache check, recursive resolver query, and root DNS server resolution.
   - **Relevance Category:** **1** (Partial gold passage overlap + Substantively sufficient answer evidence).

2. **Chunk `1/3`** (Original Rank 2, Score: 0.7795; Skipped in Diversified)
   - **Vector ID:** `...-page-1-chunk-3` (355 chars raw / 261 chars normalized)
   - **Stored Text:**
     > *"(usually provided by your ISP). 4. Resolver asks the root DNS server → then the TLD server (like .com) → then the authoritative DNS server . 5. The final IP address is returned to your computer. 6. Now, your browser knows where to send the request. For Example:"*
   - **Comparison & Evidence:**
     - Overlaps with `1/2`, but has 0 literal overlap with Gold Chunk `1/1`.
     - Starts mid-sentence as a continuation fragment. While it lists steps 4–6 (TLD server, authoritative server, IP return), standalone it is an incomplete sentence fragment.
   - **Relevance Category:** **0** (Conservative standalone judgment; see Sensitivity Analysis for lenient 1).

3. **Chunk `2/0`** (Original Rank 3, Score: 0.7247; Diversified Rank 2)
   - **Vector ID:** `...-page-2-chunk-0` (790 chars raw / 590 chars normalized)
   - **Stored Text:**
     > *"Imagine you want to call Pizza Hut but don’t know the number. You check your contact list (cache), then try Google (recursive resolver), which checks various directories until it gets the number. Now that your computer knows the IP address , it’s like having the phone number of a friend. But before you start talking (or exchanging data), you need to establish a connection, just like how we say 'hello' and wait for the other person to respond on a call. This brings us to our second step: the TCP Handshake . 2. TCP Handshake: Establishing a Reliable Connection TCP (Transmission Control"*
   - **Comparison & Evidence:**
     - 0 overlap with Gold Chunk `1/1`.
     - Uses a Pizza Hut phone call analogy and transitions into TCP Handshake. Does not contain the technical DNS translation process.
   - **Relevance Category:** **0** (Insufficient answer evidence).

4. **Chunk `1/0`** (Diversified Rank 3; Original Rank 5, Score: 0.7043)
   - **Vector ID:** `...-page-1-chunk-0` (793 chars raw / 591 chars normalized)
   - **Stored Text:**
     > *"BASIC UNDERSTANDING OF APIs Have you ever wondered how you're able to access information from anywhere in the world—instantly? You simply type www.google.com , and within seconds, the search bar is ready. Or maybe you type codingninjas.com , and in a blink, you're on our website. But stop and think: How does this actually happen? What magical steps take place behind the scenes to bring a website from some server in another corner of the world, right into your screen? It may feel like magic, but it’s actually the power of the Internet Protocol Stack working in harmony So the very first"*
   - **Comparison & Evidence:**
     - Overlaps with the introductory clause of Gold Chunk `1/1` (*"actually the power of the Internet Protocol Stack working in harmony So the very first"*).
     - Contains only introductory rhetorical hooks; does not mention DNS or IP address translation.
   - **Relevance Category:** **0** (No answer evidence).

---

### Question 2: Which messages make up the TCP three-way handshake?
- **Verified Gold Coordinate:** Page 2, Chunk 1 (`2/1`)
- **Verified Gold Passage Text (`2/1`):**
  > *"TCP Handshake . 2. TCP Handshake: Establishing a Reliable Connection TCP (Transmission Control Protocol) ensures that data is sent reliably and in order . Before any actual data transfer, a connection must be established using the three-way handshake . The 3-Step Handshake: 1. SYN (Synchronize) – Client says, “Hey, I want to talk. Here's my sequence number.” 2. SYN-ACK – Server replies, “Sure! Here’s my sequence number. Acknowledging yours too.” 3. ACK (Acknowledge) – Client says, “Cool, ready to start!” Only after this process can data transfer begin. For Example: Think of it like a polite phone call:"*

#### Retrieved Candidate Chunks (Q2):

1. **Chunk `2/1`** (Original Rank 1, Score: 0.7770; Diversified Rank 1)
   - **Vector ID:** `...-page-2-chunk-1` (797 chars raw / 609 chars normalized)
   - **Stored Text:** Exact verbatim match to the verified gold passage.
   - **Comparison & Evidence:**
     - Identifies all three messages explicitly: `1. SYN (Synchronize)`, `2. SYN-ACK`, and `3. ACK (Acknowledge)` with definitions.
   - **Relevance Category:** **2** (Complete gold passage present).

2. **Chunk `2/2`** (Original Rank 2, Score: 0.7470; Skipped in Diversified)
   - **Vector ID:** `...-page-2-chunk-2` (791 chars raw / 571 chars normalized)
   - **Stored Text:**
     > *"process can data transfer begin. For Example: Think of it like a polite phone call: ● You dial a number (SYN) ● They pick up and say, 'Hello?' (SYN-ACK) ● You say, “Hi, can we talk now?” (ACK) ● Now the conversation starts. Great! So now both your browser and the server are ready to talk. The connection has been established. Now comes the actual conversation— the data transfer . Think of it like you're now ready to send a large letter, but it won’t fit in one envelope. So what do we do? We split it into parts and send them one by one. That’s exactly what happens in"*
   - **Comparison & Evidence:**
     - Overlaps with Gold Chunk `2/1` tail (*"process can data transfer begin. For Example: Think of it like a polite phone call:"*).
     - Directly names all three handshake messages: `SYN`, `SYN-ACK`, and `ACK`.
   - **Relevance Category:** **1** (Partial gold passage overlap + Substantively sufficient answer evidence).

3. **Chunk `6/0`** (Original Rank 3, Score: 0.7052; Diversified Rank 2)
   - **Vector ID:** `...-page-6-chunk-0` (796 chars raw / 458 chars normalized)
   - **Stored Text:**
     > *"The 4-Step FIN-ACK Process: 1. Client sends FIN – 'I’m done sending data.' 2. Server replies with ACK – 'Got it.' 3. Server sends its own FIN – 'I'm also done.' 4. Client replies with ACK – 'Okay, all done!' The connection is now closed cleanly. For Example: You and your friend say goodbye over a phone call: ● 'Okay, I’m done.' ● 'Got it. I’m done too.' ● 'Cool. Bye!' Notes: ● Connections not properly closed may stay in the TIME_WAIT state. ● TCP ensures"*
   - **Comparison & Evidence:**
     - 0 overlap with Gold Chunk `2/1`.
     - Describes connection termination (`FIN-ACK`), not connection establishment (`three-way handshake`).
   - **Relevance Category:** **0** (Insufficient answer evidence).

4. **Chunk `4/2`** (Diversified Rank 3; Original Rank 4, Score: 0.6984)
   - **Vector ID:** `...-page-4-chunk-2` (533 chars raw / 401 chars normalized)
   - **Stored Text:**
     > *"sequence number, source/destination) and payload (data). ● TCP ensures all packets arrive and are in the correct order. ● Acknowledgments (ACKs) are sent to confirm receipt. For Example: Imagine sending a long handwritten letter by post, but your envelope can only carry 1 page at a time. You number each page (sequence number), and your friend confirms receipt of each one. Example in a Web Scenario:"*
   - **Comparison & Evidence:**
     - 0 overlap with Gold Chunk `2/1`.
     - Discusses packet-level acknowledgments during data transmission, not the three-way handshake messages.
   - **Relevance Category:** **0** (Insufficient answer evidence).

---

### Question 3: How do serverless APIs differ from serverful APIs?
- **Verified Gold Coordinate:** Page 8, Chunk 3 (`8/3`)
- **Verified Gold Passage Text (`8/3`):**
  > *"single day? How does it scale to serve thousands of users at the same time? Let’s now look behind the scenes — into the modern architecture that powers these APIs. The server (which is the brain behind the API) can be hosted in two major ways: Serverful APIs — Traditional, always-on infrastructure Serverless APIs — Modern, event-driven, cloud-native infrastructure Each plays a crucial role depending on the nature of the task and system , and both are used to serve LLMs, sometimes in combination."*

#### Retrieved Candidate Chunks (Q3):

1. **Chunk `9/0`** (Original Rank 1, Score: 0.8146; Diversified Rank 1)
   - **Vector ID:** `...-page-9-chunk-0` (793 chars raw / 585 chars normalized)
   - **Stored Text:**
     > *"Let’s break them down. 1. What is a Serverful API? A Serverful API is hosted on a dedicated server or virtual machine (VM) . This server is always running , whether or not any request is being made. Think of it like a restaurant kitchen that’s open 24x7 — even when no one is ordering. In this model: ● You (or the provider) must set up the infrastructure : the machine, memory, CPU, scaling, etc. ● The server is always available to take requests, like an always-open shop. ● All LLM computation (e.g., generating embeddings or responses) happens on this machine. LLM Example: Using a"*
   - **Comparison & Evidence:**
     - 0 overlap with Gold Chunk `8/3`.
     - Explains Serverful APIs in isolation; does not mention Serverless or explain the differences.
   - **Relevance Category:** **0** (Incomplete; does not provide the requested contrast).

2. **Chunk `10/0`** (Original Rank 2, Score: 0.7864; Diversified Rank 2)
   - **Vector ID:** `...-page-10-chunk-0` (800 chars raw / 596 chars normalized)
   - **Stored Text:**
     > *"2. What is a Serverless API? A Serverless API means: You write your logic, deploy it to the cloud, and the cloud handles everything else. There’s still a server behind the scenes, but you never manage it . The server spins up on demand , executes your code (like processing a prompt), then shuts down. You only pay for the time used , not for idle waiting. LLM Example: Calling Gemini or ChatGPT APIs When you call Gemini LLM using the GEMINI SDK, you’re using a serverless endpoint . Google’s infrastructure automatically: ● Spins up LLM inference engines, ● Loads the model weights, ● Processes"*
   - **Comparison & Evidence:**
     - 0 overlap with Gold Chunk `8/3`.
     - Explicitly articulates the fundamental differences: unmanaged vs managed servers, on-demand spin-up vs always-on, pay-for-time-used vs pay for idle waiting.
   - **Relevance Category:** **1** (Substantively sufficient answer evidence).

3. **Chunk `10/1`** (Original Rank 3, Score: 0.7735; Skipped in Diversified)
   - **Vector ID:** `...-page-10-chunk-1` (796 chars raw / 592 chars normalized)
   - **Stored Text:**
     > *"automatically: ● Spins up LLM inference engines, ● Loads the model weights, ● Processes your prompt, ● Sends back the output, ● And tears down the computer — all within milliseconds. You didn’t manage the server — you just called the API. Advantages of Serverless APIs: ● Zero infrastructure management — just write code. ● Cost-effective — pay only for usage. ● Elastic scalability — handles 1 or 1 million LLM calls without you doing anything. ● Ideal for event-driven systems (e.g., AI chatbots, summarization tools, etc.) Disadvantages: ● Cold starts — slight delay if the function hasn’t"*
   - **Comparison & Evidence:**
     - 0 overlap with Gold Chunk `8/3`.
     - Details the operational differences: zero infrastructure management, pay-per-use, elastic scalability, event-driven nature, and cold starts.
   - **Relevance Category:** **1** (Substantively sufficient answer evidence).

4. **Chunk `9/2`** (Diversified Rank 3; Original Rank 5, Score: 0.7644)
   - **Vector ID:** `...-page-9-chunk-2` (794 chars raw / 590 chars normalized)
   - **Stored Text:**
     > *"latency because the server is always warm. ● Good for LLMs fine-tuned for specific tasks running on enterprise hardware. Disadvantages: ● High cost: You pay even when no one uses the API. ● Poor scalability: Handling spikes in LLM usage requires manual scaling or expensive autoscaling setups. ● Higher responsibility: You manage security patches, infra failures, and updates. But what if you don’t want to manage any servers ? What if you want your code to just run automatically when a request comes , and not waste money or effort on idle machines? That’s where Serverless APIs come in —"*
   - **Comparison & Evidence:**
     - 0 overlap with Gold Chunk `8/3`.
     - Directly contrasts Serverful drawbacks (high idle costs, manual scaling, maintenance overhead) with Serverless advantages (automated on-demand execution without server management).
   - **Relevance Category:** **1** (Substantively sufficient answer evidence).

---

### Question 4: What does HTTP 403 Forbidden mean?
- **Verified Gold Coordinate:** Page 16, Chunk 0 (`16/0`) [Nominal]
- **Verified Gold Passage Text (`16/0`):**
  > *"● 200 OK → The request was successful, and the server is returning the requested data. ● 201 Created → A new resource was successfully created (used in POST requests). 3xx – Redirection These codes indicate that further action is needed to complete the request. ● 301 Moved Permanently → The requested page or resource has been moved to a new URL. ● 302 Found → Temporarily redirected to another resource. 4xx – Client Errors These codes mean there was something wrong with the client’s request (e.g., bad input, unauthorized access). ● 400 Bad Request → The request was malformed or missing"*

> [!IMPORTANT]
> **Key Finding on Gold Label Misalignment:**  
> The nominal gold coordinate `16/0` supplied in early benchmarks stops at `400 Bad Request`. **`16/0` does not contain the definition of HTTP 403 Forbidden.**  
> The actual definition of HTTP 403 Forbidden is contained in **Chunk `16/1`**, which was retrieved at **Rank 3 in Original Top-3** and promoted to **Rank 2 in Diversified Top-3**.

#### Retrieved Candidate Chunks (Q4):

1. **Chunk `15/1`** (Original Rank 1, Score: 0.7336; Diversified Rank 1)
   - **Vector ID:** `...-page-15-chunk-1` (788 chars raw / 574 chars normalized)
   - **Stored Text:** General definition of status codes and 1xx codes. No mention of 403.
   - **Relevance Category:** **0** (Insufficient answer evidence).

2. **Chunk `15/2`** (Original Rank 2, Score: 0.7239; Skipped in Diversified)
   - **Vector ID:** `...-page-15-chunk-2` (438 chars raw / 330 chars normalized)
   - **Stored Text:** Covers 1xx Informational and 2xx Success codes. No mention of 403.
   - **Relevance Category:** **0** (Insufficient answer evidence).

3. **Chunk `16/1`** (Original Rank 3, Score: 0.7145; Diversified Rank 2)
   - **Vector ID:** `...-page-16-chunk-1` (788 chars raw / 586 chars normalized)
   - **Stored Text:**
     > *"bad input, unauthorized access). ● 400 Bad Request → The request was malformed or missing required data. ● 401 Unauthorized → You need to provide a valid API key or credentials. ● 403 Forbidden → You don’t have permission to access this resource. ● 404 Not Found → The requested resource doesn’t exist. 5xx – Server Errors These codes indicate that something went wrong on the server’s side , not yours. ● 500 Internal Server Error → The server had an unexpected issue. ● 503 Service Unavailable → The server is currently overloaded or under maintenance. The content above gives you the"*
   - **Comparison & Evidence:**
     - Overlaps with the tail of nominal Gold Chunk `16/0` (*"bad input, unauthorized access). ● 400 Bad Request → The request was malformed or missing required data."*).
     - **Directly provides the exact, authoritative answer to Q4:**  
       `● 403 Forbidden → You don’t have permission to access this resource.`
   - **Relevance Category:** **1** (Partial gold chunk overlap + Definitive answer evidence).

4. **Chunk `13/2`** (Diversified Rank 3; Original Rank 5, Score: 0.6903)
   - **Vector ID:** `...-page-13-chunk-2` (709 chars raw / 529 chars normalized)
   - **Stored Text:** Overview of HTTP protocol structure and postal analogy. No status code definitions.
   - **Relevance Category:** **0** (Insufficient answer evidence).

---

### Question 5: What does an HTTP status code tell a client about its request?
- **Verified Gold Coordinate:** Page 15, Chunk 1 (`15/1`)
- **Verified Gold Passage Text (`15/1`):**
  > *"Did it fail? Did we make a mistake? Or did something go wrong on the server? That’s where HTTP Status Codes come in. What is an HTTP Status Code? Whenever a client (like your browser or Python script) sends a request to a server, the server responds with more than just the data — It also sends a three-digit status code that tells you what happened to your request. These codes help you understand: ● Whether the request was successful ● If there was an error ● Or if some other action is needed Types of HTTP Status Codes: 1xx – Informational These codes indicate that the"*

#### Retrieved Candidate Chunks (Q5):

1. **Chunk `15/1`** (Original Rank 1, Score: 0.8050; Diversified Rank 1)
   - **Vector ID:** `...-page-15-chunk-1` (788 chars raw / 574 chars normalized)
   - **Stored Text:** Exact verbatim match to verified gold passage.
   - **Comparison & Evidence:**
     - Directly defines the role of status codes: *"tells you what happened to your request... help you understand: Whether the request was successful, If there was an error, Or if some other action is needed"*.
   - **Relevance Category:** **2** (Complete gold passage present).

2. **Chunk `15/2`** (Original Rank 2, Score: 0.7722; Skipped in Diversified)
   - **Vector ID:** `...-page-15-chunk-2` (438 chars raw / 330 chars normalized)
   - **Stored Text:**
     > *"is needed Types of HTTP Status Codes: 1xx – Informational These codes indicate that the request has been received and is still being processed. ● 100 Continue → The server has received the request headers and is waiting for the body. 2xx – Success These codes mean the request was successfully received, understood, and processed."*
   - **Comparison & Evidence:**
     - Overlaps with Gold Chunk `15/1` tail (*"is needed Types of HTTP Status Codes: 1xx – Informational These codes indicate that the"*).
     - Explains 1xx and 2xx class meanings, but lacks the overarching definition of what status codes communicate.
   - **Relevance Category:** **0** (Conservative standalone judgment; see Sensitivity Analysis for lenient 1).

3. **Chunk `16/0`** (Original Rank 3, Score: 0.7458; Diversified Rank 2)
   - **Vector ID:** `...-page-16-chunk-0` (793 chars raw / 591 chars normalized)
   - **Stored Text:** Lists individual status codes (200, 201, 301, 302, 400).
   - **Comparison & Evidence:** Provides individual status code examples, but does not define the overarching purpose of HTTP status codes.
   - **Relevance Category:** **0** (Insufficient answer evidence).

4. **Chunk `13/2`** (Diversified Rank 3; Original Rank 4, Score: 0.7439)
   - **Vector ID:** `...-page-13-chunk-2` (709 chars raw / 529 chars normalized)
   - **Stored Text:** HTTP protocol introduction. Does not explain status codes.
   - **Relevance Category:** **0** (Insufficient answer evidence).

---

## 5. Step 4: Metric Formulations & Recalculation

### 5.1 Mathematical Definitions

1. **Binary Relevance ($rel \in \{0, 1\}$):**
   $$rel_i = \begin{cases} 1 & \text{if Category} \ge 1 \\ 0 & \text{if Category} = 0 \end{cases}$$

2. **Precision@3:**
   $$\text{Precision@3} = \frac{1}{3} \sum_{i=1}^3 rel_i$$

3. **Recall@3:**
   Given that each question has at least one verified answer in the document:
   $$\text{Recall@3} = \begin{cases} 1.0 & \text{if } \sum_{i=1}^3 rel_i \ge 1 \\ 0.0 & \text{otherwise} \end{cases}$$

4. **Mean Reciprocal Rank (MRR@3):**
   $$\text{RR@3} = \begin{cases} \frac{1}{\text{rank}^*} & \text{where } \text{rank}^* = \min \{ i \mid rel_i = 1 \} \\ 0 & \text{if no relevant chunk in Top-3} \end{cases}$$
   $$\text{MRR@3} = \frac{1}{|Q|} \sum_{q \in Q} \text{RR@3}_q$$

5. **Discounted Cumulative Gain & nDCG@3:**
   - **Binary nDCG@3:** Uses binary relevance $rel_i \in \{0, 1\}$:
     $$\text{DCG}_{\text{bin}}@3 = \sum_{i=1}^3 \frac{rel_i}{\log_2(i + 1)}$$
     $$\text{nDCG}_{\text{bin}}@3 = \frac{\text{DCG}_{\text{bin}}@3}{\text{IDCG}_{\text{bin}}@3}$$
     *(where $\text{IDCG}_{\text{bin}}@3$ is calculated by sorting the retrieved binary relevance scores descending).*
   - **Graded nDCG@3 (Standard Exponential Formulation):** Uses graded relevance $g_i \in \{0, 1, 2\}$:
     $$\text{DCG}_{\text{graded}}@3 = \sum_{i=1}^3 \frac{2^{g_i} - 1}{\log_2(i + 1)}$$
     $$\text{nDCG}_{\text{graded}}@3 = \frac{\text{DCG}_{\text{graded}}@3}{\text{IDCG}_{\text{graded}}@3}$$

---

### 5.2 Primary Metric Results (Standard Conservative Judgment)

#### Original Top-3 (Live Retrieval Ranking)

| Question | Top-3 Coords | Graded Labels | Binary Labels | Precision@3 | Recall@3 | Reciprocal Rank | nDCG@3 (Binary) | nDCG@3 (Graded) |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| **Q1 — DNS** | `1/2, 1/3, 2/0` | `[1, 0, 0]` | `[1, 0, 0]` | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Q2 — TCP handshake** | `2/1, 2/2, 6/0` | `[2, 1, 0]` | `[1, 1, 0]` | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Q3 — Serverless APIs** | `9/0, 10/0, 10/1` | `[0, 1, 1]` | `[0, 1, 1]` | 0.667 | 1.000 | 0.500 | 0.693 | 0.693 |
| **Q4 — HTTP 403** | `15/1, 15/2, 16/1`| `[0, 0, 1]` | `[0, 0, 1]` | 0.333 | 1.000 | 0.333 | 0.500 | 0.500 |
| **Q5 — HTTP status codes**| `15/1, 15/2, 16/0`| `[2, 0, 0]` | `[1, 0, 0]` | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Macro Average** | — | — | — | **0.467** | **1.000** | **0.767** | **0.839** | **0.839** |

#### Diversified Top-3 (Corrected Adjacency Deduplication)

| Question | Top-3 Coords | Graded Labels | Binary Labels | Precision@3 | Recall@3 | Reciprocal Rank | nDCG@3 (Binary) | nDCG@3 (Graded) |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| **Q1 — DNS** | `1/2, 2/0, 1/0` | `[1, 0, 0]` | `[1, 0, 0]` | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Q2 — TCP handshake** | `2/1, 6/0, 4/2` | `[2, 0, 0]` | `[1, 0, 0]` | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Q3 — Serverless APIs** | `9/0, 10/0, 9/2`  | `[0, 1, 1]` | `[0, 1, 1]` | 0.667 | 1.000 | 0.500 | 0.693 | 0.693 |
| **Q4 — HTTP 403** | `15/1, 16/1, 13/2`| `[0, 1, 0]` | `[0, 1, 0]` | 0.333 | 1.000 | 0.500 | 0.631 | 0.631 |
| **Q5 — HTTP status codes**| `15/1, 16/0, 13/2`| `[2, 0, 0]` | `[1, 0, 0]` | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Macro Average** | — | — | — | **0.400** | **1.000** | **0.800** | **0.865** | **0.865** |

---

### 5.3 Sensitivity Analysis: Lenient Neighbor-Overlap Scoring

If neighbor chunks providing supporting evidence/range definitions (`1/3` for Q1 and `15/2` for Q5) are granted Category 1:

| Metric | Original Top-3 (Lenient) | Diversified Top-3 (Lenient) | Delta |
|---|:---:|:---:|:---:|
| **Precision@3** | 0.600 | 0.400 | -0.200 (redundancy removal) |
| **Recall@3** | 1.000 | 1.000 | 0.000 |
| **MRR@3** | 0.767 | 0.800 | **+0.033** |
| **nDCG@3 (Binary)** | 0.839 | 0.865 | **+0.026** |
| **nDCG@3 (Graded)** | 0.839 | 0.865 | **+0.026** |

The core conclusion is invariant across both conservative and lenient interpretations: Diversified retrieval increases MRR and nDCG while eliminating redundant adjacent chunks.

---

## 6. Comparison: Original vs. Diversified Top-3

### 6.1 Head-to-Head Metric Summary

| Metric | Previous Coordinate Proxy | Original Top-3 (Text-Level) | Diversified Top-3 (Text-Level) | Text-Level Net Change |
|---|:---:|:---:|:---:|:---:|
| **Precision@3** | 0.133 | **0.467** | **0.400** | -0.067 |
| **Recall@3** | 0.400 | **1.000** | **1.000** | **0.000 (100% coverage)** |
| **MRR@3** | 0.400 | **0.767** | **0.800** | **+0.033 (+4.3%)** |
| **nDCG@3 (Binary)**| *N/A* | **0.839** | **0.865** | **+0.026 (+3.1%)** |
| **nDCG@3 (Graded)**| *N/A* | **0.839** | **0.865** | **+0.026 (+3.1%)** |

### 6.2 Key Retrieval Mechanism Observations

1. **Why MRR Improved under Diversification:**
   In Original retrieval for **Q4**, the candidate ranking was `15/1` (Rank 1), `15/2` (Rank 2), and `16/1` (Rank 3). The true answer passage (`16/1`) sat at Rank 3, yielding an RR of `1/3 = 0.333`.
   The diversification rule detected that `15/2` shared page 15 with selected chunk `15/1` and differed by index $\le 1$, correctly skipping `15/2`. This promoted `16/1` to **Rank 2**, increasing RR to `1/2 = 0.500`.

2. **Deduplication vs. Top-3 Precision Trade-off:**
   In Original retrieval, Q2 had two relevant chunks (`2/1` and `2/2`) because `2/2` repeated the SYN/SYN-ACK/ACK sequence in an example. The diversification heuristic skipped `2/2` and selected `6/0` (FIN-ACK) and `4/2` (packet sequencing). This reduced redundant answers within Top-3 (P@3 for Q2 dropping from 0.667 to 0.333), while preserving 100% Recall and freeing downstream LLM context for broader coverage.

3. **Consistent Semantic Diversity in Q3:**
   In Q3, Original retrieval included adjacent chunks `10/0` and `10/1`. Diversification skipped `10/1` and admitted `9/2`. Because `9/2` also contains substantive contrast evidence between serverful and serverless models, Q3 retained **0.667 Precision@3** and **0.500 MRR@3** under both rankings.

---

## 7. Limitations & Discussion

1. **Chunk Boundary Discrepancies:**
   The active Pinecone index contains 50 vectors, whereas the current local chunker produces 42 chunks. As confirmed in `pinecone_chunk_audit_report.md`, the indexed chunks overlap adjacent neighbors by 79–94 normalized characters. This overlap means answer evidence is frequently partitioned or duplicated across chunk boundaries (e.g., DNS translation across `1/1`, `1/2`, and `1/3`).
2. **Gold Label Granularity:**
   Gold coordinates in legacy benchmarks were assigned as single point-labels. In reality, answers to multi-step technical questions (e.g., DNS resolution or Serverful vs Serverless architecture) span multiple chunks.
3. **Small Query Sample Size:**
   The evaluation is conducted on five benchmark questions. While these findings demonstrate the superiority of text-level auditing over coordinate matching, wider query evaluation is recommended before altering production retrieval thresholds.

---

## 8. Safety Verification Checklist

- **No Pinecone upsert, update, or delete:** Verified. All vector store interactions were read-only `/vectors/fetch` calls.
- **No document ingestion or re-indexing:** Verified.
- **No Gemini, Groq, OpenAI, or LLM API calls:** Verified. No embedding vectors or chat completions were generated.
- **No source code, configuration, or `.env` modifications:** Verified.
- **No dependencies installed or upgraded:** Verified.
- **No deployment or GitHub push:** Verified.
- **No credential exposure:** Verified. All API keys and host connection strings were kept confidential.

---
*Report compiled on 2026-11-13.*
