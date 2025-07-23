package lab.demo_ai;

import lombok.extern.slf4j.Slf4j;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;

import java.util.HashMap;
import java.util.Map;

@Controller
@Slf4j
public class MainController {

    @Autowired
    ApplicationContext ac;

    @PostMapping("/api/main")
    public ResponseEntity<Map<String, Object>> mainController(@RequestBody Map<String, Object> inParam){
        Map<String,Object> resultMap = new HashMap<>();

        log.info("inParam : " + inParam);
        baseTx account = (baseTx) ac.getBean("account");
        Map<String, Object> param = (Map<String, Object>) inParam.get("payload");
        resultMap = account.execute(param);


        return ResponseEntity.ok(resultMap);
    }
}
