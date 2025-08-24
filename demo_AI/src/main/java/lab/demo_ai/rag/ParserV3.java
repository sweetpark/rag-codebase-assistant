package lab.demo_ai.rag;



import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;
import java.util.stream.Collectors;


public class ParserV3 {

    public static void main(String[] args) throws Exception {
        Map<String, String> cli = parseArgs(args);
        Path root = Paths.get(cli.getOrDefault("--root", "./src/main"));
        System.out.println(root.toString());
        Path out  = Paths.get(cli.getOrDefault("--out", "RESULT_PYTHON/3_THIRD/chunk.jsonl"));
        Files.createDirectories(out.getParent());

        Rules rules = Rules.defaultRules();
        ChunkWriter writer = new ChunkWriter(out);

        // 1) Walk project tree
        List<Path> files = Files.walk(root)
                .filter(Files::isRegularFile)
                .filter(p -> rules.include(p.toString()))
                .collect(Collectors.toList());

        // Index for linking
        Map<String, Chunk> byId = new LinkedHashMap<>();
        Map<String, Chunk> bySymbol = new HashMap<>(); // e.g., Class.method
        Map<String, Chunk> bySqlId  = new HashMap<>();
        Map<String, List<Chunk>> byRoute = new HashMap<>(); // method+path -> controller chunks

        JavaExtractor javaExtractor = new JavaExtractor(rules);
        XmlExtractor  xmlExtractor  = new XmlExtractor(rules);
        HtmlExtractor htmlExtractor = new HtmlExtractor(rules);
        Linker linker               = new Linker();

        // 2) Extract chunks per file
        for (Path p : files) {
            String pathStr = root.relativize(p).toString().replace('\\','/');
            String text = Files.readString(p, StandardCharsets.UTF_8);

            if (pathStr.endsWith(".java")) {
                List<Chunk> chunks = javaExtractor.extractJava(pathStr, text);
                for (Chunk c : chunks) { indexAndWrite(c, writer, byId, bySymbol, bySqlId, byRoute); }
            } else if (pathStr.endsWith(".xml")) {
                List<Chunk> chunks = xmlExtractor.extractMyBatis(pathStr, text);
                for (Chunk c : chunks) { indexAndWrite(c, writer, byId, bySymbol, bySqlId, byRoute); }
            } else if (pathStr.endsWith(".html") || pathStr.endsWith(".jsp")) {
                List<Chunk> chunks = htmlExtractor.extractHtml(pathStr, text);
                for (Chunk c : chunks) { indexAndWrite(c, writer, byId, bySymbol, bySqlId, byRoute); }
            }
        }

        // 3) Linking pass (augment existing chunks)
        linker.linkControllerToService(bySymbol, byId);
        linker.linkServiceToMapper(bySymbol, bySqlId, byId);
        linker.linkViewToController(byRoute, byId);

        // 4) Rewrite augmented chunks
        writer.rewriteAll(byId.values());

        System.out.println("Done -> " + out);
    }

    static void indexAndWrite(Chunk c, ChunkWriter w,
                              Map<String, Chunk> byId,
                              Map<String, Chunk> bySymbol,
                              Map<String, Chunk> bySqlId,
                              Map<String, List<Chunk>> byRoute) throws IOException {
        if (c == null) return;
        byId.put(c.id, c);
        if (c.symbol != null) bySymbol.put(c.symbol, c);
        if (c.sqlId != null) bySqlId.put(c.sqlId, c);
        if (c.httpMethod != null && c.httpPath != null) {
            String key = c.httpMethod + " " + c.httpPath;
            byRoute.computeIfAbsent(key, k -> new ArrayList<>()).add(c);
        }
        w.append(c); // initial write; will be rewritten after linking
    }

    static Map<String, String> parseArgs(String[] args) {
        Map<String, String> m = new HashMap<>();
        for (int i=0;i<args.length;i+=2) {
            if (i+1 < args.length) m.put(args[i], args[i+1]);
        }
        return m;
    }

}


// ----------------------------- Rules -----------------------------
class Rules {
    final Pattern includeJava = Pattern.compile("src/(main|webapp)/.*\\.java$|.*\\.java$");
    final Pattern includeXml  = Pattern.compile(".*(mappers|mapper|mybatis).*\\.xml$|.*Mapper\\.xml$|.*\\.xml$");
    final Pattern includeHtml = Pattern.compile(".*(templates|WEB-INF|views).*\\.(html|jsp)$|.*\\.(html|jsp)$");
    final Set<String> excludeDirs = Set.of("target/", "build/", "node_modules/", "test/");

    boolean include(String relPath) {
        String p = relPath.replace('\\','/');
        for (String ex : excludeDirs) if (p.contains("/"+ex) || p.startsWith(ex)) return false;
        return includeJava.matcher(p).find() || includeXml.matcher(p).find() || includeHtml.matcher(p).find();
    }

    static Rules defaultRules(){ return new Rules(); }
}

// ----------------------------- Data Model -----------------------------
class Chunk {
    String id;                     // unique
    String project = "default";
    String layer;                  // controller|service|mapper|view
    String language;               // java|xml|html|jsp
    String filePath;
    String symbol;                 // Class.method for Java
    String httpMethod;             // GET|POST|PUT|DELETE
    String httpPath;               // /api/... from @RequestMapping
    String sqlId;                  // MyBatis id
    String sqlKind;                // select|insert|update|delete
    List<String> tables = new ArrayList<>();
    List<String> calls = new ArrayList<>();      // controller -> service symbols
    List<String> dependsOn = new ArrayList<>();  // service -> mapper sqlId
    List<String> mapsToView = new ArrayList<>(); // controller -> view file(s)
    List<String> mapsToController = new ArrayList<>(); // view -> controller key(method path)

    String summary;                // auto summarized text
    String content;                // trimmed snippet

    static String makeId(String... parts){
        return String.join("-", parts).replaceAll("[^a-zA-Z0-9_-]","_");
    }
}

// ----------------------------- Writer -----------------------------
class ChunkWriter {
    private final Path out;
    private final ObjectMapper om = new ObjectMapper();
    ChunkWriter(Path out){ this.out = out; }

    void append(Chunk c) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8,
                Files.exists(out)? StandardOpenOption.APPEND: StandardOpenOption.CREATE)) {
            w.write(toJson(c));
            w.newLine();
        }
    }

    void rewriteAll(Collection<Chunk> chunks) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            for (Chunk c : chunks) { w.write(toJson(c)); w.newLine(); }
        }
    }

    String toJson(Chunk c) throws IOException {
        ObjectNode n = om.createObjectNode();
        n.put("id", c.id);
        n.put("project", c.project);
        n.put("layer", c.layer);
        n.put("language", c.language);
        n.put("file_path", c.filePath);
        if (c.symbol!=null) n.put("symbol", c.symbol);
        if (c.httpMethod!=null) n.put("http_method", c.httpMethod);
        if (c.httpPath!=null) n.put("http_path", c.httpPath);
        if (c.sqlId!=null) n.put("sql_id", c.sqlId);
        if (c.sqlKind!=null) n.put("sql_kind", c.sqlKind);
        if (!c.tables.isEmpty()) n.putPOJO("tables", c.tables);
        if (!c.calls.isEmpty()) n.putPOJO("calls", c.calls);
        if (!c.dependsOn.isEmpty()) n.putPOJO("depends_on", c.dependsOn);
        if (!c.mapsToView.isEmpty()) n.putPOJO("maps_to_view", c.mapsToView);
        if (!c.mapsToController.isEmpty()) n.putPOJO("maps_to_controller", c.mapsToController);
        if (c.summary!=null) n.put("summary", c.summary);
        if (c.content!=null) n.put("content", c.content);
        return n.toString();
    }
}

// ----------------------------- Java Extractor -----------------------------
class JavaExtractor {
    final Rules rules;
    final JavaParser parser;
    public JavaExtractor(Rules rules){
        this.rules = rules;
        ParserConfiguration cfg = new ParserConfiguration();
        this.parser = new JavaParser(cfg);
    }

    List<Chunk> extractJava(String path, String code){
        List<Chunk> out = new ArrayList<>();
        Optional<CompilationUnit> cu = parser.parse(code).getResult();
        if (cu.isEmpty()) return out;

        for (ClassOrInterfaceDeclaration cls : cu.get().findAll(ClassOrInterfaceDeclaration.class)) {
            String className = cls.getNameAsString();
            boolean isController = hasAnyAnnotation(cls, Set.of("Controller","RestController"));
            boolean isService    = hasAnyAnnotation(cls, Set.of("Service"));

            // Collect injected field types for linking (Service/Mapper variables)
            Map<String,String> fieldTypeByVar = new HashMap<>();
            for (FieldDeclaration f : cls.findAll(FieldDeclaration.class)) {
                f.getVariables().forEach(v -> {
                    String var = v.getNameAsString();
                    String type = f.getElementType().isClassOrInterfaceType() ?
                            f.getElementType().asClassOrInterfaceType().getNameAsString(): f.getElementType().toString();
                    fieldTypeByVar.put(var, type);
                });
            }

            for (MethodDeclaration m : cls.getMethods()) {
                if (isController && hasRequestMapping(m)) {
                    Chunk c = new Chunk();
                    c.layer = "controller"; c.language="java"; c.filePath=path;
                    c.symbol = className+"."+m.getNameAsString();
                    String[] route = readRoute(m);
                    c.httpMethod = route[0]; c.httpPath = route[1];
                    c.id = Chunk.makeId("controller", c.symbol);
                    c.summary = Summarizer.controllerSummary(m, c.httpMethod, c.httpPath);
                    c.content = trim(m.toString());
                    // service calls from body: serviceVar.method()
                    Set<String> calls = new LinkedHashSet<>();
                    m.findAll(MethodCallExpr.class).forEach(call -> {
                        call.getScope().ifPresent(scope -> {
                            String sv = scope.toString();
                            if (fieldTypeByVar.containsKey(sv)) {
                                // symbol: ServiceClass.method
                                calls.add(fieldTypeByVar.get(sv) + "." + call.getNameAsString());
                            }
                        });
                    });
                    c.calls.addAll(calls);
                    out.add(c);
                }
                if (isService && m.isPublic()) {
                    Chunk c = new Chunk();
                    c.layer = "service"; c.language="java"; c.filePath=path;
                    c.symbol = className+"."+m.getNameAsString();
                    c.id = Chunk.makeId("service", c.symbol);
                    c.summary = Summarizer.serviceSummary(m);
                    c.content = trim(m.toString());

                    // mapper calls: mapperVar.method()
                    Set<String> mapperIds = new LinkedHashSet<>();
                    m.findAll(MethodCallExpr.class).forEach(call -> {
                        call.getScope().ifPresent(scope -> {
                            String var = scope.toString();
                            if (fieldTypeByVar.containsKey(var) && fieldTypeByVar.get(var).toLowerCase().contains("mapper")) {
                                mapperIds.add(call.getNameAsString()); // method name matches <select id>
                            }
                        });
                    });
                    c.dependsOn.addAll(mapperIds); // temporary, resolved in linker
                    out.add(c);
                }
            }
        }
        return out;
    }

    static boolean hasAnyAnnotation(ClassOrInterfaceDeclaration cls, Set<String> names){
        for (AnnotationExpr a : cls.getAnnotations()) {
            if (names.contains(a.getNameAsString())) return true;
        }
        return false;
    }
    static boolean hasRequestMapping(MethodDeclaration m){
        for (AnnotationExpr a : m.getAnnotations()) {
            String n = a.getNameAsString();
            if (n.endsWith("Mapping") || n.equals("RequestMapping")) return true;
        }
        return false;
    }
    static String[] readRoute(MethodDeclaration m){
        String method = "GET"; String path = "/";
        for (AnnotationExpr a : m.getAnnotations()) {
            String n = a.getNameAsString();
            if (n.equals("GetMapping")) method = "GET";
            else if (n.equals("PostMapping")) method = "POST";
            else if (n.equals("PutMapping")) method = "PUT";
            else if (n.equals("DeleteMapping")) method = "DELETE";
            else if (n.equals("RequestMapping")) method = "GET"; // default fallback
            String s = a.toString();
            Matcher m1 = Pattern.compile("\"([^\"]+)\"").matcher(s);
            if (m1.find()) path = m1.group(1);
        }
        return new String[]{method, path};
    }
    static String trim(String s){ return s.length()>2000? s.substring(0,2000)+"\n/*...trimmed*/" : s; }
}

// ----------------------------- XML Extractor -----------------------------
class XmlExtractor {
    final Rules rules;
    final Pattern tablePat = Pattern.compile("(?i)(?:from|update|into)\\s+([a-z0-9_]+)");
    public XmlExtractor(Rules rules){ this.rules = rules; }

    List<Chunk> extractMyBatis(String path, String xml){
        List<Chunk> out = new ArrayList<>();
        Document doc = Jsoup.parse(xml, "", org.jsoup.parser.Parser.xmlParser());
        for (Element e : doc.select("select,insert,update,delete")) {
            Chunk c = new Chunk();
            c.layer = "mapper"; c.language="xml"; c.filePath=path;
            c.sqlId = e.attr("id");
            c.sqlKind = e.tagName();
            c.id = Chunk.makeId("mapper", c.sqlId);
            String sql = e.text();
            c.content = sql.length()>2000? sql.substring(0,2000)+"\n/*...trimmed*/" : sql;
            c.summary = Summarizer.sqlSummary(c.sqlKind, sql);
            Matcher m = tablePat.matcher(sql);
            while (m.find()) c.tables.add(m.group(1));
            out.add(c);
        }
        return out;
    }
}

// ----------------------------- HTML/JSP Extractor -----------------------------
class HtmlExtractor {
    final Rules rules;
    final Pattern fetchUrl = Pattern.compile("fetch\\(\\s*['\"]([^'\"]+)['\"]");
    final Pattern ajaxUrl  = Pattern.compile("\\$\\.ajax\\s*\\(\\s*\\{[^}]*url\\s*:\\s*['\"]([^'\"]+)['\"]");
    public HtmlExtractor(Rules rules){ this.rules = rules; }

    List<Chunk> extractHtml(String path, String html){
        List<Chunk> out = new ArrayList<>();
        Document doc = Jsoup.parse(html);
        Chunk c = new Chunk();
        c.layer = "view"; c.language = path.endsWith(".jsp")? "jsp":"html"; c.filePath=path;
        c.id = Chunk.makeId("view", path.replace('/','_'));
        c.summary = Summarizer.viewSummary(doc);
        c.content = keyDom(doc);

        // forms
        for (Element f : doc.select("form")) {
            String method = Optional.ofNullable(f.attr("method")).orElse("GET").toUpperCase();
            String action = Optional.ofNullable(f.attr("action")).orElse("");
            if (!action.isBlank()) c.mapsToController.add(method+" "+action);
        }
        // fetch / $.ajax urls
        String scriptText = doc.select("script").stream().map(Element::data).collect(Collectors.joining("\n"));
        Matcher m1 = fetchUrl.matcher(scriptText); while (m1.find()) c.mapsToController.add("GET "+m1.group(1));
        Matcher m2 = ajaxUrl.matcher(scriptText);  while (m2.find()) c.mapsToController.add("POST "+m2.group(1));

        out.add(c);
        return out;
    }

    static String keyDom(Document d){
        // keep IDs/classes hints only (small snippet)
        Elements ids = d.select("*[id]");
        Elements inputs = d.select("input,select,button");
        StringBuilder sb = new StringBuilder();
        sb.append("<!-- key DOM ids/classes -->\n");
        ids.stream().limit(50).forEach(e -> sb.append("#").append(e.id()).append(" ").append(e.tagName()).append("\n"));
        sb.append("<!-- inputs/buttons -->\n");
        inputs.stream().limit(50).forEach(e -> sb.append(e.tagName()).append("[name=").append(e.attr("name")).append("] id=").append(e.id()).append("\n"));
        return sb.toString();
    }
}

// ----------------------------- Linker -----------------------------
class Linker {
    // After initial extraction, resolve placeholders into concrete references
    void linkControllerToService(Map<String, Chunk> bySymbol, Map<String, Chunk> byId){
        for (Chunk c : byId.values()) if ("controller".equals(c.layer)) {
            List<String> resolved = new ArrayList<>();
            for (String sym : c.calls) if (bySymbol.containsKey(sym)) resolved.add(bySymbol.get(sym).id);
            c.calls = resolved; // replace symbols with chunk ids
        }
    }
    void linkServiceToMapper(Map<String, Chunk> bySymbol, Map<String, Chunk> bySqlId, Map<String, Chunk> byId){
        for (Chunk c : byId.values()) if ("service".equals(c.layer)) {
            List<String> resolved = new ArrayList<>();
            for (String sqlId : c.dependsOn) if (bySqlId.containsKey(sqlId)) resolved.add(bySqlId.get(sqlId).id);
            c.dependsOn = resolved;
        }
    }
    void linkViewToController(Map<String, List<Chunk>> byRoute, Map<String, Chunk> byId){
        for (Chunk c : byId.values()) if ("view".equals(c.layer)) {
            List<String> resolved = new ArrayList<>();
            for (String key : c.mapsToController) {
                if (byRoute.containsKey(key)) {
                    for (Chunk ctrl : byRoute.get(key)) resolved.add(ctrl.id);
                }
            }
            c.mapsToController = resolved;
        }
    }
}

// ----------------------------- Summarizer -----------------------------
class Summarizer {
    static String controllerSummary(MethodDeclaration m, String httpMethod, String httpPath){
        return String.format("%s %s 요청 처리. 입력 파라미터/바인딩을 통해 서비스 호출 후 결과 반환.", httpMethod, httpPath);
    }
    static String serviceSummary(MethodDeclaration m){
        return String.format("서비스 메서드 %s: 도메인 로직 수행 및 Mapper 연동.", m.getNameAsString());
    }
    static String sqlSummary(String kind, String sql){
        String order = sql.contains("order by") || sql.contains("ORDER BY")? "정렬 포함" : "";
        return ("SQL("+kind+") 추출. " + order).trim();
    }
    static String viewSummary(Document d){
        return "뷰 템플릿 추출: 폼/버튼/비동기 호출 URL을 분석하여 컨트롤러 라우트와 매핑.";
    }
}

/* -------------------- Pseudo build.gradle (place in your build) --------------------
plugins { id 'java' }
repositories { mavenCentral() }
dependencies {
    implementation 'com.github.javaparser:javaparser-core:3.26.1'
    implementation 'org.jsoup:jsoup:1.17.2'
    implementation 'com.fasterxml.jackson.core:jackson-databind:2.17.0'
}
application { mainClass = 'rag.Main' }
----------------------------------------------------------------------------------- */


