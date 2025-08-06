package lab.demo_ai.rag;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.NodeList;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

public class DocumentProcessor2 {
    private static final ObjectMapper mapper = new ObjectMapper();
    private static BufferedWriter writer;

    public static void main(String[] args) throws Exception {

        String rootDir = "src/main";
        String outputFile = "output/chunk.jsonl";
        writer = new BufferedWriter(new FileWriter(outputFile, false));
        Files.walk(Paths.get(rootDir))
                .filter(Files::isRegularFile)
                .forEach(DocumentProcessor2::processFile);
        writer.close();
        System.out.println("Document preprocessing complete: " + outputFile);
    }

    private static void processFile(Path path) {
        String file = path.toString().toLowerCase();
        try {
            if (file.endsWith(".html") || file.endsWith(".jsp")) {
                processHtml(path.toFile());
            } else if (file.endsWith(".java")) {
                processJava(path.toFile());
            } else if (file.endsWith(".xml")) {
                processXml(path.toFile());
            } else if (file.endsWith(".sql")) {
                processSql(path.toFile());
            }
        } catch (Exception e) {
            System.err.println("Failed to process " + path + ": " + e.getMessage());
        }
    }

    // HTML: 전체 뷰 + 스타일 블록
    private static void processHtml(File file) throws Exception {
        Document doc = Jsoup.parse(file, "UTF-8");
        // 전체 뷰
        emitChunk(file.getPath(), "HTML_VIEW", doc.html());
        // style 블록
        Elements styles = doc.select("style");
        int idx = 0;
        for (Element style : styles) {
            emitChunk(file.getPath() + "#STYLE" + (idx++), "STYLE", style.html());
        }
    }

    // Java: 메서드 단위
    private static void processJava(File file) throws Exception {
        String src = new String(Files.readAllBytes(file.toPath()));
        CompilationUnit cu = StaticJavaParser.parse(src);
        for (MethodDeclaration m : cu.findAll(MethodDeclaration.class)) {
            String id = file.getPath() + "#METHOD#" + m.getNameAsString();
            emitChunk(id, "JAVA_METHOD", m.toString());
        }
    }

    // MyBatis XML: 각 SQL 태그
    private static void processXml(File file) throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        org.w3c.dom.Document xml = builder.parse(new FileInputStream(file));
        String[] tags = {"select", "insert", "update", "delete"};
        for (String tag : tags) {
            NodeList nodes = xml.getElementsByTagName(tag);
            for (int i = 0; i < nodes.getLength(); i++) {
                String idAttr = nodes.item(i).getAttributes().getNamedItem("id") != null ?
                        nodes.item(i).getAttributes().getNamedItem("id").getTextContent() : String.valueOf(i);
                String role = "MYBATIS_" + tag.toUpperCase();
                String chunkId = file.getPath() + "#" + tag + "#" + idAttr;
                String text = nodes.item(i).getTextContent().trim();
                emitChunk(chunkId, role, text);
            }
        }
    }

    // SQL: split by ';'
    private static void processSql(File file) throws Exception {
        String sql = new String(Files.readAllBytes(file.toPath()));
        String[] parts = sql.split(";");
        for (int i = 0; i < parts.length; i++) {
            String stmt = parts[i].trim();
            if (stmt.isEmpty()) continue;
            String id = file.getPath() + "#SQL#" + i;
            emitChunk(id, "SQL_STATEMENT", stmt + ";");
        }
    }

    private static void emitChunk(String id, String role, String text) throws Exception {
        ObjectNode node = mapper.createObjectNode();
        node.put("id", id);
        node.put("role", role);
        node.put("text", text);
        writer.write(mapper.writeValueAsString(node));
        writer.newLine();
    }
}
