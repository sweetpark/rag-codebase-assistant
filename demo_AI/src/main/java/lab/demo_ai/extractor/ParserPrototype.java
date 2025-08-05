package lab.demo_ai.extractor;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.comments.JavadocComment;
import com.github.javaparser.ast.expr.MethodCallExpr;
import net.sf.jsqlparser.parser.CCJSqlParserUtil;
import net.sf.jsqlparser.statement.Statement;
import net.sf.jsqlparser.statement.select.Select;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.io.FileInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

public class ParserPrototype {
    public static void main(String[] args) throws Exception {
        String rootDir = args.length > 0 ? args[0] : ".";
        Files.walk(Paths.get(rootDir))
                .filter(Files::isRegularFile)
                .forEach(ParserPrototype::parseFile);
    }

    private static void parseFile(Path path) {
        String file = path.toString().toLowerCase();
        try {
            if (file.endsWith(".html") || file.endsWith(".jsp")) {
                parseHtml(path.toFile());
            } else if (file.endsWith(".java")) {
                parseJava(path.toFile());
            } else if (file.endsWith(".sql")) {
                parseSql(path.toFile());
            } else if (file.endsWith(".xml")) {
                parseXml(path.toFile());
            }
        } catch (Exception e) {
            System.err.println("Failed to parse " + path + ": " + e.getMessage());
        }
    }

    // HTML parser: extract data-service and data-oper
    private static void parseHtml(File file) throws Exception {
        Document doc = Jsoup.parse(file, "UTF-8");
        Elements elems = doc.select("[data-service][data-oper]");
        if (!elems.isEmpty()) {
            List<String> actions = elems.stream()
                    .map(el -> String.format("service=%s, oper=%s",
                            el.attr("data-service"), el.attr("data-oper")))
                    .collect(Collectors.toList());
            System.out.println("[HTML] " + file + " -> " + actions);
        }
    }

    // Java parser: extract methods without Javadoc and service.execute flows
    private static void parseJava(File file) throws Exception {
        String src = new String(Files.readAllBytes(file.toPath()));
        CompilationUnit cu = StaticJavaParser.parse(src);

        List<String> methods = cu.findAll(MethodDeclaration.class).stream()
                .filter(m -> !m.getComment().filter(c -> c instanceof JavadocComment).isPresent())
                .map(m -> m.getDeclarationAsString(false, false, false))
                .collect(Collectors.toList());

        List<String> flows = cu.findAll(MethodCallExpr.class).stream()
                .filter(call -> "execute".equals(call.getNameAsString()) && call.getScope().isPresent())
                .map(call -> call.toString())
                .collect(Collectors.toList());

        if (!methods.isEmpty() || !flows.isEmpty()) {
            System.out.println("[JAVA] " + file + " Methods:" + methods.size() + " Flows:" + flows.size());
        }
    }

    // SQL parser: extract SELECT/INSERT/UPDATE/DELETE
    private static void parseSql(File file) throws Exception {
        String sql = new String(Files.readAllBytes(file.toPath()));
        String[] statements = sql.split(";");
        for (String stmt : statements) {
            String trimmed = stmt.trim();
            if (trimmed.isEmpty()) continue;
            try {
                Statement parsed = CCJSqlParserUtil.parse(trimmed);
                String type = parsed.getClass().getSimpleName();
                System.out.println("[SQL] " + file + " -> " + type + ": " + trimmed.replaceAll("\s+", " ") );
            } catch (Exception ex) {
                // ignore non-SQL
            }
        }
    }

    // XML parser: extract SQL inside MyBatis tags
    private static void parseXml(File file) throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        org.w3c.dom.Document xml = builder.parse(new FileInputStream(file));
        org.w3c.dom.NodeList selects = xml.getElementsByTagName("select");
        org.w3c.dom.NodeList inserts = xml.getElementsByTagName("insert");
        org.w3c.dom.NodeList updates = xml.getElementsByTagName("update");
        org.w3c.dom.NodeList deletes = xml.getElementsByTagName("delete");
        int count = selects.getLength() + inserts.getLength() + updates.getLength() + deletes.getLength();
        if (count > 0) {
            System.out.println("[XML] " + file + " -> Statements:" + count);
        }
    }
}
