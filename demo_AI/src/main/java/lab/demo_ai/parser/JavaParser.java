package lab.demo_ai.parser;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.utils.SourceRoot;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicInteger;

public class JavaParser {

    public static void main(String[] args) throws IOException {
        // 1) 소스 루트 디렉터리 지정 (보통 src/main/java)
        Path projectRoot = Paths.get("src/main/java");
        SourceRoot sourceRoot = new SourceRoot(projectRoot);

        // 2) 빈 문자열 패키지 경로("")부터 시작해 모든 .java 파일을 파싱
        sourceRoot.parse("", (localPath, absolutePath, result) -> {
            result.ifSuccessful(cu -> {
                System.out.println("Parsed: " + absolutePath);
                // cu를 이용해 AST 처리
                try {

                    CompilationUnit unit = StaticJavaParser.parse(absolutePath);


                    // 2. 구조 정보
                    // 패키지
                    unit.getPackageDeclaration()
                            .ifPresent(pd -> System.out.println("Package: " + pd.getName()));

                    // 클래스/인터페이스
                    unit.findAll(ClassOrInterfaceDeclaration.class).forEach(c -> {
                        System.out.println("Class: " + c.getName()
                                + " (interface=" + c.isInterface() + ")");
                        // 애노테이션
                        c.getAnnotations().forEach(a ->
                                System.out.println("  Annotation: @" + a.getName()));
                    });

                    // 필드
                    unit.findAll(FieldDeclaration.class).forEach(f -> {
                        f.getVariables().forEach(v ->
                                System.out.println("Field: " + v.getName()
                                        + " : " + f.getElementType()));
                    });

                    // 메서드
                    unit.findAll(MethodDeclaration.class).forEach(m -> {
                        System.out.println("Method: " + m.getName());
                    });

                    // 3. 시그니처·시맨틱
                    unit.findAll(MethodDeclaration.class).forEach(m -> {
                        System.out.printf("  %s(%s) : %s%n",
                                m.getName(),
                                // 파라미터
                                m.getParameters().stream()
                                        .map(p -> p.getType() + " " + p.getName())
                                        .reduce((a,b)->a+", "+b).orElse(""),
                                // 반환 타입
                                m.getType());
                        // 예외
                        m.getThrownExceptions().forEach(ex ->
                                System.out.println("    throws: " + ex));
                        // Javadoc
                        m.getJavadocComment()
                                .ifPresent(j -> System.out.println("    Javadoc: " + j.getContent()));
                    });

                    // 4. 호출 관계·DI 포인트
                    unit.findAll(MethodCallExpr.class).forEach(call ->
                            System.out.println("Call: " + call.getName()
                                    + " at line " + call.getRange().map(r->r.begin.line).orElse(-1)));
                    unit.findAll(FieldDeclaration.class).stream()
                            .filter(f -> f.isAnnotationPresent("Autowired"))
                            .forEach(f -> System.out.println("DI point: " + f.getVariables().get(0).getName()));

                    // 5. 코드 메트릭
                    AtomicInteger loc = new AtomicInteger();
                    unit.getAllContainedComments().forEach(c -> {/* 주석 제외 카운트 필요하면 처리 */});
                    // 단순 LOC: 파일 전체 라인 수
                    long totalLines = Files.readAllLines(absolutePath).size();
                    System.out.println("LOC: " + totalLines);
                    // 메서드별 파라미터 개수·길이
                    unit.findAll(MethodDeclaration.class).forEach(m -> {
                        int params = m.getParameters().size();
                        int bodyLines = m.getBody()
                                .map(b -> b.getStatements().size())
                                .orElse(0);
                        System.out.printf("  %s → params=%d, statements=%d%n",
                                m.getName(), params, bodyLines);
                    });

                    // (추가) 복잡도: cyclomatic complexity 추정용 if/for/switch 개수
                    unit.findAll(MethodDeclaration.class).forEach(m -> {
                        long cc = m.findAll(Statement.class).stream()
                                .filter(s -> s.isIfStmt() || s.isForStmt() || s.isSwitchStmt() || s.isWhileStmt())
                                .count() + 1;
                        System.out.println("  " + m.getName() + " CC≈" + cc);
                    });




                } catch (IOException e) {
                    throw new RuntimeException(e);
                }


            });
            // 파일 수정 후 저장하지 않으려면 DONT_SAVE
            return SourceRoot.Callback.Result.DONT_SAVE;
        });

    }
}
