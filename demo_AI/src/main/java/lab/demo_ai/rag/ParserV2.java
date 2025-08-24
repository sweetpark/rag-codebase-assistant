package lab.demo_ai.rag;


import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.comments.JavadocComment;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ParserV2 {
    private static final ObjectMapper mapper = new ObjectMapper();

    public static void main(String[] args) throws IOException {
        // 소스 디렉토리 및 출력 파일 설정
        Path javaRoot = Paths.get("./src/main/java");
        Path htmlRoot = Paths.get("./src/main/resources/static");
        Path xmlRoot  = Paths.get("./src/main/resources/mapper");
        Path output   = Paths.get("RESULT_PYTHON/2_SECOND/chunk.jsonl");

        try (BufferedWriter writer = Files.newBufferedWriter(output)) {
            // Java 파일 파싱
            collectFiles(javaRoot, ".java").forEach(p -> parseJavaFile(p, writer));
            // HTML 파일 파싱
            collectFiles(htmlRoot, ".html").forEach(p -> parseHtml(p.toFile(), writer));
            // XML 파일 파싱
            collectFiles(xmlRoot, ".xml").forEach(p -> parseXml(p.toFile(), writer));
        }
    }

    private static List<Path> collectFiles(Path root, String ext) throws IOException {
        List<Path> list = new ArrayList<>();
        if (!Files.exists(root)) return list;
        Files.walk(root)
                .filter(p -> p.toString().endsWith(ext))
                .forEach(list::add);
        return list;
    }

    private static void parseJavaFile(Path path, BufferedWriter writer) {
        try {
            CompilationUnit cu = StaticJavaParser.parse(path);
            String filePath = path.toString();
            String pkg = cu.getPackageDeclaration()
                    .map(pd -> pd.getName().asString())
                    .orElse("");

            for (ClassOrInterfaceDeclaration clazz : cu.findAll(ClassOrInterfaceDeclaration.class)) {
                String fullClass = pkg.isEmpty() ? clazz.getNameAsString()
                        : pkg + "." + clazz.getNameAsString();

                for (MethodDeclaration method : clazz.findAll(MethodDeclaration.class)) {
                    String id      = fullClass + "." + method.getNameAsString();
                    String content = method.toString();
                    String doc     = method.getJavadocComment()
                            .map(JavadocComment::getContent)
                            .orElse("");

                    ObjectNode chunk = mapper.createObjectNode();
                    chunk.put("id", id);
                    chunk.put("file_path", filePath);
                    chunk.put("lang", "java");
                    chunk.put("type", "method");
                    chunk.put("content", content);
                    chunk.put("docstring", doc);
                    chunk.putArray("dependencies");
                    String commitMsg = getLastCommitMessage(path,
                            method.getBegin().get().line,
                            method.getEnd().get().line
                    );
                    chunk.put("commit_message", commitMsg);
                    chunk.putArray("tags").add("java");

                    writer.write(chunk.toString());
                    writer.newLine();
                }
            }
        } catch (Exception e) {
            System.err.println("[ERROR] parseJavaFile " + path + " : " + e.getMessage());
        }
    }

    private static void parseHtml(File file, BufferedWriter writer) {
        try {
            Document doc = Jsoup.parse(file, "UTF-8");

            // === 1) 전체 VIEW chunk ===
            emitChunk(writer,
                    file.toPath(),
                    file.getName() + "#VIEW",
                    file.getPath(),
                    "html",
                    "HTML_VIEW",
                    doc.html(),
                    "",
                    "html", "view");

            // === 2) <style> 블록 chunk 분리 ===
            Elements styles = doc.select("style");
            int idx = 0;
            for (Element style : styles) {
                emitChunk(writer,
                        file.toPath(),
                        file.getName() + "#STYLE" + idx,
                        file.getPath() + "#STYLE" + idx,
                        "css",
                        "STYLE",
                        style.html(),
                        "",
                        "css", "style");
                idx++;
            }

            // === 3) <script> 내 JS 함수 정규식 추출 ===
            for (Element script : doc.getElementsByTag("script")) {
                String js = script.html();
                // function foo(...) { ... } 패턴만 1차 수집 (백틱/화살표 함수는 별도 처리 필요)
                Pattern funcPattern = Pattern.compile(
                        "function\\s+(\\w+)\\s*\\(([^)]*)\\)\\s*\\{([\\s\\S]*?)\\}",
                        Pattern.MULTILINE
                );
                Matcher matcher = funcPattern.matcher(js);

                while (matcher.find()) {
                    String fnName   = matcher.group(1);
                    String fnParams = matcher.group(2);
                    String fnBody   = matcher.group(3);
                    String content  = "function " + fnName + "(" + fnParams + ") {" + fnBody + "}";

                    emitChunk(writer,
                            file.toPath(),
                            file.getName() + "#FUNC." + fnName,
                            file.getPath(),
                            "javascript",
                            "function",
                            content,
                            "",
                            "js", "function");
                }
            }
        } catch (Exception e) {
            System.err.println("[ERROR] parseHtml " + file + " : " + e.getMessage());
        }
    }

    private static void parseXml(File file, BufferedWriter writer) {
        try {
            Document xml = Jsoup.parse(
                    file, "UTF-8", "", org.jsoup.parser.Parser.xmlParser()
            );

            for (String tag : Arrays.asList("select","insert","update","delete")) {
                for (Element el : xml.getElementsByTag(tag)) {
                    String id      = file.getName() + "#" + el.attr("id");
                    String sql     = el.text().trim();
                    String comment = "";
                    if (el.previousSibling() instanceof org.jsoup.nodes.Comment) {
                        comment = ((org.jsoup.nodes.Comment)el.previousSibling()).getData();
                    }

                    ObjectNode chunk = mapper.createObjectNode();
                    chunk.put("id", id);
                    chunk.put("file_path", file.getPath());
                    chunk.put("lang", "xml");
                    chunk.put("type", "sql");
                    chunk.put("content", sql);
                    chunk.put("docstring", comment);
                    int line = el.siblingIndex();
                    String commitMsg = getLastCommitMessage(
                            file.toPath(), line, line
                    );
                    chunk.put("commit_message", commitMsg);
                    chunk.putArray("tags").add("sql");

                    writer.write(chunk.toString());
                    writer.newLine();
                }
            }
        } catch (Exception e) {
            System.err.println("[ERROR] parseXml " + file + " : " + e.getMessage());
        }
    }

    /** 공통 chunk write 헬퍼 */
    private static void emitChunk(BufferedWriter writer,
                                  Path commitPath,
                                  String id,
                                  String filePath,
                                  String lang,
                                  String type,
                                  String content,
                                  String docstring,
                                  String... tags) throws IOException {
        ObjectNode chunk = mapper.createObjectNode();
        chunk.put("id", id);
        chunk.put("file_path", filePath);
        chunk.put("lang", lang);
        chunk.put("type", type);
        chunk.put("content", content);
        chunk.put("docstring", docstring == null ? "" : docstring);
        String commitMsg = getLastCommitMessage(commitPath, 1, 1);
        chunk.put("commit_message", commitMsg);
        if (tags != null && tags.length > 0) {
            var arr = chunk.putArray("tags");
            for (String t : tags) arr.add(t);
        } else {
            chunk.putArray("tags");
        }
        writer.write(chunk.toString());
        writer.newLine();
    }

    private static String getLastCommitMessage(Path path, int startLine, int endLine) {
        try {
            ProcessBuilder pb = new ProcessBuilder(
                    "git", "log", "-1", "-L",
                    startLine + "," + endLine + ":" + path.toString()
            );
            pb.redirectErrorStream(true);
            Process p = pb.start();

            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(p.getInputStream())
            );
            String line;
            boolean inMsg = false;
            StringBuilder sb = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("    ")) inMsg = true;
                if (inMsg) sb.append(line.trim()).append(" ");
            }
            return sb.toString().trim();
        } catch (IOException e) {
            return "";
        }
    }
}
