# ai_project



# 1단계
<pre>
[샘플 API 코드 (JSP/Java + MyBatis XML)]
           │
   (1) 코드 전처리 요청
           ▼
[전처리 서비스]
(AST: JavaParser ← JSP 컴파일러(Jasper) + jsoup + XML 파서)
           │
   (2) 코드 청크 → LLM
           ▼
[LLM 주석/커밋 메시지 서비스]
(파인튜닝된 사내 모델)
           │
   (3) 주석 삽입 & 메시지 제안
           ▼
[Git Hook Runner]
(pre-commit / prepare-commit-msg)
           │
      (4) 개발자 워크스페이스
  → 주석 달린 코드 커밋
  
</pre>


# 2단계
<pre>
  [PG 시스템]  
    │  (1) MCP 프로듀서로 이벤트 발행  
    ▼
[Kafka Cluster]  
    │  (2) 토픽에 버퍼링  
    ▼
[Streaming 처리 엔진]  
(Flink/Spark)  
    │  (3) 이벤트 필터링·전처리 → 피처 및 원문(로그·릴리즈 노트 등) 추출  
    ▼
[Vector DB]  
(예: Pinecone)  
    │  (4) 문장·코드 청크 임베딩 저장  
    ▼
[문서 생성 서비스 (RAG)]  
    │  (5) 사용자 질의 → 유사 문서 검색 → LLM 호출 → 문서 생성  
    ▼
[UI/Chatbot/API]  
(운영자·개발자 인터페이스)  

</pre>
