package lab.demo_ai.parser;

import lombok.SneakyThrows;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class XmlDomParser {
    @SneakyThrows
    public static void main(String[] args) {
        Path mapperDir = Paths.get("src/main/resources/mapper");
        try (Stream<Path> paths = Files.walk(mapperDir)) {
            paths
                    .filter(p -> p.toString().endsWith(".xml"))
                    .forEach(p -> processMapperXml(p.toFile()));
        }
    }


    private static void processMapperXml(File file) {
        try {
            // 1. DOM 파서 준비
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            factory.setNamespaceAware(true);          // 네임스페이스 사용 시
            factory.setIgnoringComments(true);        // <!-- 주석 --> 무시
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document doc = builder.parse(file);

            // 2. 최상위 <mapper> 요소
            Element mapper = doc.getDocumentElement();
            String namespace = mapper.getAttribute("namespace");
            System.out.println("=== " + file.getName() + " (" + namespace + ") ===");

            // 3. CRUD 태그 순회
            NodeList nodes = mapper.getChildNodes();
            for (int i = 0; i < nodes.getLength(); i++) {
                Node n = nodes.item(i);
                if (n.getNodeType() != Node.ELEMENT_NODE) continue;
                Element e = (Element) n;
                String tag = e.getTagName();  // select, insert, update, delete 등
                String id  = e.getAttribute("id");
                String paramType  = e.getAttribute("parameterType");
                String resultType = e.getAttribute("resultType");
                String rawSql = e.getTextContent().trim().replaceAll("\\s+", " ");

                System.out.printf(
                        "%s id=\"%s\" param=\"%s\" result=\"%s\" → %s%n",
                        tag, id, paramType, resultType, rawSql
                );
            }
        } catch (Exception ex) {
            System.err.println("Failed to parse " + file + ": " + ex.getMessage());
        }
    }
}
