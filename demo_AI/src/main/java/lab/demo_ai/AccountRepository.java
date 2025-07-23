package lab.demo_ai;

import org.apache.ibatis.annotations.Mapper;

import java.util.List;
import java.util.Map;

@Mapper
public interface AccountRepository {
    List<Map<String,Object>> sltAccountInfo(Map<String, Object> param);
    Map<String,Object> sltAccountOne(Map<String, Object> param);
    void istAccount(Map<String, Object> param);
    void uptAccount(Map<String, Object> param);
    void resetPw(Map<String, Object>param);
    Map<String, Object> dupCheck(Map<String, Object> param);
}
