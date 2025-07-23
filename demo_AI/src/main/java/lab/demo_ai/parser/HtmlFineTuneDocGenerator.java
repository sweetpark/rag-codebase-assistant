package lab.demo_ai.parser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.htmlcleaner.HtmlCleaner;
import org.htmlcleaner.TagNode;
import org.htmlcleaner.XPatherException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

public class HtmlFineTuneDocGenerator {

    private static final String UI_ROOT = "src/main/resources/static/ui";
    private final HtmlCleaner cleaner;
    private final ObjectMapper mapper;

    public HtmlFineTuneDocGenerator() {
        this.cleaner = new HtmlCleaner();
        cleaner.getProperties().setTranslateSpecialEntities(true);
        cleaner.getProperties().setOmitComments(true);
        this.mapper = new ObjectMapper();
    }

    public static void main(String[] args) throws IOException {
        HtmlFineTuneDocGenerator generator = new HtmlFineTuneDocGenerator();
        generator.processAllHtml();
    }

    private void processAllHtml() throws IOException {
        Path uiDir = Paths.get(UI_ROOT);
        try (Stream<Path> paths = Files.walk(uiDir)) {
            paths.filter(p -> p.toString().endsWith(".html"))
                    .forEach(this::processFile);
        }
    }

    private void processFile(Path htmlPath) {
        try {
            TagNode root = cleaner.clean(htmlPath.toFile());
            ObjectNode docJson = mapper.createObjectNode();
            docJson.put("filePath", htmlPath.toString());

            // 1. 문서 메타데이터
            extractMetadata(root, docJson);
            // 2. 구조·계층 정보
            extractStructure(root, docJson);
            // 3. 콘텐츠 정보
            extractContent(root, docJson);
            // 4. 속성·스타일 정보
            extractAttributes(root, docJson);
            // 5. 상호작용 요소
            extractInteraction(root, docJson);
            // 6. 접근성·SEO 정보
            extractAccessibility(root, docJson);
            // 7. 코드 메트릭
            extractMetrics(root, htmlPath, docJson);

            extractLabels(root, docJson);
            extractForms(root, docJson);
            extractEventHandlers(root, docJson);



            // JSON 출력
            String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(docJson);
            System.out.println(json);
        } catch (Exception e) {
            System.err.println("Failed to process " + htmlPath + ": " + e.getMessage());
        }
    }

    private void extractMetadata(TagNode root, ObjectNode doc) throws XPatherException {
        ObjectNode meta = mapper.createObjectNode();
        // title
        Object[] titles = root.evaluateXPath("//title");
        if (titles.length > 0) {
            meta.put("title", ((TagNode) titles[0]).getText().toString());
        }
        // meta description, keywords
        extractMetaTag(root, meta, "description");
        extractMetaTag(root, meta, "keywords");
        // open graph
        extractMetaProperty(root, meta, "og:title");
        extractMetaProperty(root, meta, "og:description");
        // canonical link
        Object[] canon = root.evaluateXPath("//link[@rel='canonical']");
        if (canon.length > 0) {
            meta.put("canonical", ((TagNode) canon[0]).getAttributeByName("href"));
        }
        doc.set("metadata", meta);
    }

    private void extractMetaTag(TagNode root, ObjectNode meta, String name) throws XPatherException {
        Object[] nodes = root.evaluateXPath("//meta[@name='" + name + "']");
        if (nodes.length > 0) {
            TagNode n = (TagNode) nodes[0];
            meta.put(name, n.getAttributeByName("content"));
        }
    }

    private void extractMetaProperty(TagNode root, ObjectNode meta, String prop) throws XPatherException {
        Object[] nodes = root.evaluateXPath("//meta[@property='" + prop + "']");
        if (nodes.length > 0) {
            TagNode n = (TagNode) nodes[0];
            meta.put(prop, n.getAttributeByName("content"));
        }
    }

    private void extractStructure(TagNode root, ObjectNode doc) throws XPatherException {
        ObjectNode struct = mapper.createObjectNode();
        // headings
        ArrayNode headings = mapper.createArrayNode();
        for (int level = 1; level <= 6; level++) {
            Object[] hs = root.evaluateXPath("//h" + level);
            for (Object o : hs) {
                TagNode h = (TagNode) o;
                ObjectNode hn = mapper.createObjectNode();
                hn.put("tag", "h" + level);
                hn.put("text", h.getText().toString());
                headings.add(hn);
            }
        }
        struct.set("headings", headings);
        // sections semantic
        String[] sections = {"header","nav","main","section","article","aside","footer"};
        ArrayNode secs = mapper.createArrayNode();
        for (String tag : sections) {
            Object[] arr = root.evaluateXPath("//" + tag);
            if (arr.length > 0) {
                ObjectNode sn = mapper.createObjectNode();
                sn.put("tag", tag);
                sn.put("count", arr.length);
                secs.add(sn);
            }
        }
        struct.set("sections", secs);
        doc.set("structure", struct);
    }

    private void extractContent(TagNode root, ObjectNode doc) throws XPatherException {
        ObjectNode content = mapper.createObjectNode();
        // paragraphs
        ArrayNode paras = mapper.createArrayNode();
        Object[] ps = root.evaluateXPath("//p");
        for (Object o : ps) {
            paras.add(((TagNode) o).getText().toString());
        }
        content.set("paragraphs", paras);
        // links
        ArrayNode links = mapper.createArrayNode();
        Object[] as = root.evaluateXPath("//a[@href]");
        for (Object o : as) {
            TagNode a = (TagNode) o;
            ObjectNode ln = mapper.createObjectNode();
            ln.put("text", a.getText().toString());
            ln.put("href", a.getAttributeByName("href"));
            links.add(ln);
        }
        content.set("links", links);
        // tables
        ArrayNode tables = mapper.createArrayNode();
        Object[] tbs = root.evaluateXPath("//table");
        for (Object o : tbs) {
            TagNode tb = (TagNode) o;
            Object[] rows = tb.evaluateXPath(".//tr");
            ObjectNode tbj = mapper.createObjectNode();
            tbj.put("rows", rows.length);
            // columns via first row
            if (rows.length > 0) {
                TagNode firstRow = (TagNode) rows[0];
                Object[] cols = firstRow.evaluateXPath(".//th|.//td");
                tbj.put("columns", cols.length);
            }
            tables.add(tbj);
        }
        content.set("tables", tables);
        doc.set("content", content);
    }

    private void extractAttributes(TagNode root, ObjectNode doc) throws XPatherException {
        ObjectNode attrs = mapper.createObjectNode();
        // class/id
        ArrayNode classes = mapper.createArrayNode();
        ArrayNode ids = mapper.createArrayNode();
        Object[] all = root.evaluateXPath("//*[@class or @id]");
        for (Object o : all) {
            TagNode tn = (TagNode) o;
            String cls = tn.getAttributeByName("class");
            String id = tn.getAttributeByName("id");
            if (cls != null) classes.add(cls);
            if (id  != null) ids.add(id);
        }
        attrs.set("classes", classes);
        attrs.set("ids", ids);
        // data-* attributes
        ArrayNode dataAttrs = mapper.createArrayNode();
        for (Object o : all) {
            TagNode tn = (TagNode) o;
            tn.getAttributes().forEach((k,v) -> {
                if (k.startsWith("data-")) dataAttrs.add(k + "=" + v);
            });
        }
        attrs.set("dataAttributes", dataAttrs);
        doc.set("attributes", attrs);
    }

    private void extractInteraction(TagNode root, ObjectNode doc) throws XPatherException {
        ObjectNode inter = mapper.createObjectNode();
        // forms
        ArrayNode forms = mapper.createArrayNode();
        Object[] fs = root.evaluateXPath("//form");
        for (Object o : fs) {
            TagNode f = (TagNode) o;
            ObjectNode fn = mapper.createObjectNode();
            fn.put("action", f.getAttributeByName("action"));
            fn.put("method", f.getAttributeByName("method"));
            forms.add(fn);
        }
        inter.set("forms", forms);
        // buttons
        ArrayNode btns = mapper.createArrayNode();
        Object[] bs = root.evaluateXPath("//button");
        for (Object o : bs) {
            TagNode b = (TagNode) o;
            btns.add(b.getText().toString());
        }
        inter.set("buttons", btns);
        // scripts
        ArrayNode scripts = mapper.createArrayNode();
        Object[] scrs = root.evaluateXPath("//script");
        for (Object o : scrs) {
            TagNode s = (TagNode) o;
            String src = s.getAttributeByName("src");
            scripts.add(src != null ? src : "inline");
        }
        inter.set("scripts", scripts);
        doc.set("interaction", inter);
    }

    private void extractAccessibility(TagNode root, ObjectNode doc) throws XPatherException {
        ObjectNode acc = mapper.createObjectNode();
        // aria-*
        ArrayNode aria = mapper.createArrayNode();
        Object[] nodes = root.evaluateXPath("//*[@aria-label or @aria-hidden or @role]");
        for (Object o : nodes) {
            TagNode t = (TagNode) o;
            t.getAttributes().forEach((k,v) -> {
                if (k.startsWith("aria-") || k.equals("role")) {
                    ObjectNode an = mapper.createObjectNode();
                    an.put("name", k);
                    an.put("value", v);
                    aria.add(an);
                }
            });
        }
        acc.set("aria", aria);
        doc.set("accessibility", acc);
    }

    private void extractMetrics(TagNode root, Path htmlPath, ObjectNode doc) throws IOException, XPatherException {
        ObjectNode met = mapper.createObjectNode();
        // LOC
        long totalLines = Files.readAllLines(htmlPath).size();
        met.put("loc", totalLines);
        // tag counts
        Map<String, Integer> counts = new HashMap<>();
        Object[] all = root.evaluateXPath("//*");
        for (Object o : all) {
            String tag = ((TagNode) o).getName();
            counts.merge(tag, 1, Integer::sum);
        }
        ObjectNode tagCounts = mapper.createObjectNode();
        counts.forEach(tagCounts::put);
        met.set("tagCounts", tagCounts);
        doc.set("metrics", met);
    }

    // 추가: <label> 태그 추출
    private void extractLabels(TagNode root, ObjectNode doc) throws XPatherException {
        ArrayNode labels = mapper.createArrayNode();
        for (Object o : root.evaluateXPath("//label")) {
            TagNode l = (TagNode) o;
            ObjectNode ln = mapper.createObjectNode();
            ln.put("for", l.getAttributeByName("for"));
            ln.put("text", l.getText().toString());
            labels.add(ln);
        }
        doc.set("labels", labels);
    }

    // 추가: 폼 필드 상세 추출
    private void extractForms(TagNode root, ObjectNode doc) throws XPatherException {
        ArrayNode forms = mapper.createArrayNode();
        for (Object o : root.evaluateXPath("//form")) {
            TagNode f = (TagNode) o;
            ObjectNode fn = mapper.createObjectNode();
            fn.put("action", f.getAttributeByName("action"));
            fn.put("method", f.getAttributeByName("method"));
            ArrayNode fields = mapper.createArrayNode();
            for (Object ci : f.evaluateXPath(".//input|.//select|.//textarea")) {
                TagNode field = (TagNode) ci;
                ObjectNode fi = mapper.createObjectNode();
                fi.put("tag", field.getName());
                fi.put("name", field.getAttributeByName("name"));
                fi.put("type", field.getAttributeByName("type"));
                fi.put("placeholder", field.getAttributeByName("placeholder"));
                fi.put("required", field.getAttributeByName("required")!=null);
                fields.add(fi);
            }
            fn.set("fields", fields);
            forms.add(fn);
        }
        doc.set("formsDetailed", forms);
    }
    // 추가: 이벤트 핸들러 추출
    private void extractEventHandlers(TagNode root, ObjectNode doc) throws XPatherException {
        ArrayNode events = mapper.createArrayNode();
        ObjectNode ev;
        for (Object o : root.evaluateXPath("//*[@onclick]")) {
            TagNode t = (TagNode) o;
            ev = mapper.createObjectNode();
            ev.put("tag", t.getName());
            ev.put("onclick", t.getAttributeByName("onclick"));
            events.add(ev);
        }
        doc.set("events", events);
    }
}
