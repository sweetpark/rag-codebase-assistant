package lab.demo_ai.parser;

import org.htmlcleaner.HtmlCleaner;
import org.htmlcleaner.TagNode;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class HtmlParser {
    public static void main(String[] args) throws IOException {
        Path uiDir = Paths.get("src/main/resources/static/ui");

        HtmlCleaner cleaner = new HtmlCleaner();
        cleaner.getProperties().setTranslateSpecialEntities(true);
        cleaner.getProperties().setOmitComments(true);

        try (Stream<Path> paths = Files.walk(uiDir)) {
            paths.filter(p -> p.toString().endsWith(".html"))
                    .forEach(path -> {
                        try {
                            parseHtmlFile(cleaner, path.toFile());
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    });
        }
    }


    private static void parseHtmlFile(HtmlCleaner cleaner, File file) throws IOException {
        TagNode root = cleaner.clean(file);
        traverse(root);
    }

    public static void traverse(TagNode node){
        System.out.printf("<%s> attrs=%s, text=\"%s\"%n",
                node.getName(),
                node.getAttributes(),
                node.getText().toString().trim()
        );
        for (Object child : node.getAllChildren()) {
            if (child instanceof TagNode) {
                traverse((TagNode) child);
            }
        }
    }
}
