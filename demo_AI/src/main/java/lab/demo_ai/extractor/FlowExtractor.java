package lab.demo_ai.extractor;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.StringLiteralExpr;
import com.github.javaparser.ast.stmt.IfStmt;
import org.htmlcleaner.HtmlCleaner;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

public class FlowExtractor {
    public static void main(String[] args) throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        ArrayNode resultFlows = mapper.createArrayNode();

        // 1) Parse UI flows
        List<ObjectNode> uiFlows = parseUI("src/main/resources/static/ui");

        // 2) Parse Controller
        ObjectNode controller = parseController("src/main/java/lab/demo_ai/MainController.java");

        // 3) Parse Service oper->repository methods
        Map<String, List<String>> serviceMap = parseService(
                "src/main/java/lab/demo_ai/AccountService.java",
                "execute"
        );

        // 4) Parse Mapper XML for SQL statements
        Map<String,String> sqlMap = parseMapper("src/main/resources/mapper/AccountMapper.xml");

        // 5) Assemble complete flows
        for (ObjectNode uiFlow : uiFlows) {
            ArrayNode seq = (ArrayNode) uiFlow.get("sequence");
            ArrayNode calls = mapper.createArrayNode();
            ArrayNode sqlStmts = mapper.createArrayNode();

            for (int i = 0; i < seq.size(); i++) {
                ObjectNode step = (ObjectNode) seq.get(i);
                String oper = step.has("oper") ? step.get("oper").asText() : null;
                List<String> repos = serviceMap.getOrDefault(oper, Collections.emptyList());
                for (String repoMethod : repos) {
                    ObjectNode call = mapper.createObjectNode();
                    call.put("oper", oper);
                    call.put("repositoryMethod", repoMethod);
                    calls.add(call);

                    String sql = sqlMap.getOrDefault(repoMethod, "/* SQL not found */");
                    ObjectNode s = mapper.createObjectNode();
                    s.put("namespace", "lab.demo_ai.AccountRepository");
                    s.put("id", repoMethod);
                    s.put("sql", sql);
                    sqlStmts.add(s);
                }
            }

            ObjectNode flow = mapper.createObjectNode();
            flow.set("ui", uiFlow.get("ui"));
            flow.set("controller", controller);
            // service info
            ObjectNode serviceNode = mapper.createObjectNode();
            serviceNode.put("beanName", "account");
            serviceNode.put("class", "lab.demo_ai.AccountService");
            serviceNode.put("method", "execute");
            flow.set("service", serviceNode);

            flow.set("calls", calls);
            flow.set("sqlStatements", sqlStmts);
            resultFlows.add(flow);
        }

        // Write JSON output
        Path out = Paths.get("build/fullFlows.json");
        Files.createDirectories(out.getParent());
        mapper.writerWithDefaultPrettyPrinter().writeValue(out.toFile(), resultFlows);
        System.out.println("Full flows written to " + out);
    }

    // UI Parsing
    private static List<ObjectNode> parseUI(String uiDir) throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        HtmlCleaner cleaner = new HtmlCleaner();
        Pattern sendPattern = Pattern.compile(
                "sendAjax\\(\\s*['\"](?<type>\\w+)['\"]\\s*,\\s*['\"](?<service>\\w+)['\"]\\s*,\\s*(?<data>\\w+)",
                Pattern.DOTALL
        );
        Pattern eventPattern = Pattern.compile(
                "\\$\\(\"(?<sel>[^\"]+)\"\\)\\.on\\('(?<evt>\\w+)'\\s*,\\s*function\\s*\\([^)]*\\)\\s*\\{(?<bod>.*?)\\}",
                Pattern.DOTALL
        );
        List<ObjectNode> uiFlows = new ArrayList<>();

        try (Stream<Path> paths = Files.walk(Paths.get(uiDir))) {
            paths.filter(p -> p.toString().endsWith(".html")).forEach(p -> {
                try {
                    String js = Files.readString(p, StandardCharsets.UTF_8);
                    Matcher evM = eventPattern.matcher(js);
                    Matcher sa = sendPattern.matcher(js);
                    while (evM.find() || sa.find()) {

                        String sel = null;
                        String evt = null;
                        String bod = null;
                        ObjectNode ui = mapper.createObjectNode();

                        if(evM.find()){

                            sel = evM.group("sel");
                            evt = evM.group("evt");
                            bod = evM.group("bod");
                            ui.put("selector", sel);
                            ui.put("event", evt);
                            ui.put("url", "/api/main");
                            sa = sendPattern.matcher(bod);
                        }else{
                            ui.put("selector", "inline");
                            ui.put("event", "no");
                            ui.put("url", "/api/main");
                            sa = sendPattern.matcher(js);
                            bod = js;
                        }
                        ArrayNode seq = mapper.createArrayNode();
                        while (sa.find()) {
                            ObjectNode step = mapper.createObjectNode();
                            step.put("service", sa.group("service"));
                            step.put("type", sa.group("type"));
                            // extract oper if present
                            Pattern opPat = Pattern.compile(sa.group("data")+"\\.oper\\s*=\\s*['\"](\\w+)['\"]");
                            Matcher om = opPat.matcher(bod);
                            if (om.find()) step.put("oper", om.group(1));
                            // payload fields
                            Set<String> fields = new LinkedHashSet<>();
                            Pattern pf = Pattern.compile(sa.group("data")+"\\.(\\w+)\\s*=",
                                    Pattern.DOTALL
                            );
                            Matcher fm = pf.matcher(bod);
                            while (fm.find()) fields.add(fm.group(1));
                            ArrayNode pfs = step.putArray("payloadFields");
                            fields.forEach(pfs::add);
                            seq.add(step);
                        }
                        ObjectNode flow = mapper.createObjectNode();
                        flow.set("ui", ui);
                        flow.set("sequence", seq);
                        uiFlows.add(flow);
                    }

                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            });
        }
        return uiFlows;
    }

    // Controller Parsing
    private static ObjectNode parseController(String ctrlPath) throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        CompilationUnit cu = StaticJavaParser.parse(Paths.get(ctrlPath));
        MethodDeclaration md = cu.findFirst(MethodDeclaration.class,
                m -> m.getNameAsString().equals("mainController")).orElseThrow();
        ObjectNode ctrl = mapper.createObjectNode();
        ctrl.put("class", cu.getPrimaryTypeName().orElse("MainController"));
        ctrl.put("method", md.getNameAsString());
        // annotation @PostMapping
        cu.findAll(com.github.javaparser.ast.expr.NormalAnnotationExpr.class).stream()
                .filter(a->a.getNameAsString().equals("PostMapping")).findFirst()
                .ifPresent(a -> a.getPairs().stream()
                        .filter(p->p.getNameAsString().equals("value"))
                        .findFirst().ifPresent(p-> ctrl.put("route", p.getValue().toString().replaceAll("\"", "")))
                );
        ctrl.put("paramKey", "service");
        return ctrl;
    }

    // Service Parsing
    private static Map<String,List<String>> parseService(String svcPath, String methodName) throws Exception {
        Map<String,List<String>> map = new LinkedHashMap<>();
        CompilationUnit cu = StaticJavaParser.parse(Paths.get(svcPath));
        MethodDeclaration exec = cu.findFirst(MethodDeclaration.class,
                m->m.getNameAsString().equals(methodName)).orElseThrow();
        exec.findAll(IfStmt.class).forEach(ifst-> {
            ifst.getCondition().findAll(MethodCallExpr.class).forEach(call-> {
                if(call.getNameAsString().equals("equals")
                        && call.getArgument(0) instanceof StringLiteralExpr) {
                    String oper = ((StringLiteralExpr)call.getArgument(0)).getValue();
                    List<String> repos = new ArrayList<>();
                    ifst.getThenStmt().findAll(MethodCallExpr.class).stream()
                            .filter(c->c.getScope().map(Object::toString).orElse("").equals("accountRepository"))
                            .forEach(c->repos.add(c.getNameAsString()));
                    map.put(oper, repos);
                }
            });
        });
        return map;
    }

    // Mapper XML Parsing
    private static Map<String,String> parseMapper(String xmlPath) throws Exception {
        Map<String,String> sqls = new HashMap<>();
        DocumentBuilder db = DocumentBuilderFactory.newInstance().newDocumentBuilder();
        Document doc = db.parse(xmlPath);
        Element root = doc.getDocumentElement();
        String ns = root.getAttribute("namespace");
        for(String tag: List.of("select","insert","update","delete")){
            NodeList nl = root.getElementsByTagName(tag);
            for(int i=0;i<nl.getLength();i++){
                Element e = (Element)nl.item(i);
                sqls.put(e.getAttribute("id"), e.getTextContent().trim().replaceAll("\\s+"," "));
            }
        }
        return sqls;
    }
}
