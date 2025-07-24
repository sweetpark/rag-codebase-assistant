package lab.demo_ai.extractor;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.htmlcleaner.HtmlCleaner;
import org.htmlcleaner.TagNode;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.regex.Matcher;

import java.util.regex.Pattern;
import java.util.stream.Stream;

public class UIFlowExtractor {

    // Matches $("selector").on('event', 'childSelector'? , function(){ ... })
    private static final Pattern EVENT_HANDLER = Pattern.compile(
            // \$\("selector"\)\.on\('\s*event\s*'
            "\\$\\(\"(?<selector>[^\"]+)\"\\)\\.on\\('\\s*(?<event>\\w+)\\s*'"
                    // (?:\s*,\s*'child')?\s*,\s*function\s*(…)\{body\}\s*\);
                    + "(?:\\s*,\\s*'(?<child>[^']+)')?"
                    + "\\s*,\\s*function\\s*\\([^)]*\\)\\s*\\{"
                    + "(?<body>.*?)"
                    + "\\}\\s*\\);",
            Pattern.DOTALL
    );

    // Matches sendAjax("TYPE", "SERVICE", dataVar, ...)
    private static final Pattern SEND_AJAX = Pattern.compile(
            "sendAjax\\(\\s*['\"](?<type>\\w+)['\"]\\s*,\\s*['\"](?<service>\\w+)['\"]\\s*,\\s*(?<data>\\w+)\\s*,",
            Pattern.DOTALL
    );

    private static final Pattern FUNC_DEF = Pattern.compile(
            "function\\s+(?<fname>\\w+)\\s*\\([^)]*\\)\\s*\\{(?<body>.*?)\\}",
            Pattern.DOTALL
    );


    public static void main(String[] args) throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        ArrayNode flows = mapper.createArrayNode();
        HtmlCleaner cleaner = new HtmlCleaner();

        // Traverse UI HTML files
        try (Stream<Path> paths = Files.walk(Paths.get("src/main/resources/static/ui"))) {
            paths.filter(p -> p.toString().endsWith(".html"))
                    .forEach(htmlPath -> {
                        try {
                            // Read entire file (to catch all JS)
                            String jsFull = Files.readString(htmlPath, StandardCharsets.UTF_8);
                            // Clean for delegated handlers if needed
                            TagNode root = cleaner.clean(htmlPath.toFile());

                            Matcher eh = EVENT_HANDLER.matcher(jsFull);
                            Set<String> functionsToFollow = new LinkedHashSet<>();

                            while (eh.find()) {
                                String selector = eh.group("child") != null ? eh.group("child") : eh.group("selector");
                                String event = eh.group("event");
                                String handlerBody = eh.group("body");

                                // Try inline sendAjax calls
                                ArrayNode seq = mapper.createArrayNode();
                                boolean foundInline = extractAjaxCalls(handlerBody, jsFull, mapper, seq);

                                // If no inline, record functions called
                                if (!foundInline) {
                                    Pattern callPat = Pattern.compile("(?<fname>\\w+)\\s*\\(");
                                    Matcher callM = callPat.matcher(handlerBody);
                                    while (callM.find()) {
                                        functionsToFollow.add(callM.group("fname"));
                                    }
                                }

                                // Follow named functions
                                for (String fname : functionsToFollow) {
                                    Matcher fd = FUNC_DEF.matcher(jsFull);
                                    while (fd.find()) {
                                        if (fd.group("fname").equals(fname)) {
                                            String fbody = fd.group("body");
                                            extractAjaxCalls(fbody, jsFull, mapper, seq);
                                        }
                                    }
                                }

                                // Assemble flow JSON if any calls
                                if (seq.size() > 0) {
                                    ObjectNode flow = mapper.createObjectNode();
                                    ObjectNode ui = flow.putObject("ui");
                                    ui.put("selector", selector);
                                    ui.put("event", event);
                                    ui.put("url", "/api/main");
                                    ui.set("sequence", seq);
                                    flows.add(flow);
                                }
                            }

                        } catch (Exception e) {
                            System.err.println("Failed parsing " + htmlPath + ": " + e.getMessage());
                        }
                    });
        }

        // Write output
        Path out = Paths.get("build/uiFlows.json");
        Files.createDirectories(out.getParent());
        mapper.writerWithDefaultPrettyPrinter().writeValue(out.toFile(), flows);
        System.out.println("UI flows written to " + out);
    }


    private static boolean extractAjaxCalls(String body, String fullJs, ObjectMapper mapper, ArrayNode seq) {
        Matcher am = SEND_AJAX.matcher(body);
        boolean found = false;
        while (am.find()) {
            found = true;
            String type = am.group("type");
            String service = am.group("service");
            String dataVar = am.group("data");

            // Collect payload fields
            Set<String> fields = new LinkedHashSet<>();
            Pattern pf = Pattern.compile(dataVar + "\\.(\\w+)\\s*=");
            Matcher fm = pf.matcher(body);
            while (fm.find()) {
                fields.add(fm.group(1));
            }
            ObjectNode call = mapper.createObjectNode();
            call.put("service", service);
            if (fields.contains("oper")) {
                Matcher om = Pattern.compile(dataVar + "\\.oper\\s*=\\s*['\"](\\w+)['\"]").matcher(body);
                if (om.find()) call.put("oper", om.group(1));
            }
            call.put("type", type);
            ArrayNode pfArr = call.putArray("payloadFields");
            fields.forEach(pfArr::add);
            seq.add(call);
        }
        return found;
    }

}
