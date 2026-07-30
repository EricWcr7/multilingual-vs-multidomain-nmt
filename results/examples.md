# Mandarin demonstration examples

These are a small, rule-selected excerpt. The repository does not publish bulk source text or references.
The corpus excerpts below are from NTREX-128 (CC BY-SA 4.0) at revision `468c6b69c7f6a75d31d4743d9daba2af566cc18d`.

## Corpus examples

| ID | Domain | English source | Reference | NLLB prediction |
|---|---|---|---|---|
| ntrex:1655:zho_Hans | news | Conservative backbencher Peter Bone told the march in Birmingham that the UK 'would have been out' by now if Mr Farage was Brexit Secretary. | 保守党后座议员彼得•伯恩(Peter Bone)在伯明翰向游行队伍表示,如果法拉奇担任英国退欧大臣,英国早就脱欧了。 | 保守派后卫彼得· (Peter Bone) 在伯明翰的游行中表示,如果法拉奇先生是英国脱欧部长, |
| ntrex:0044:zho_Hans | news | A shark attacked and injured a 13-year-old boy Saturday while he was diving for lobster in California on the opening day of lobster season, officials said. | 官员表示,周六,一名 13 岁男童在加州的龙虾节开放日潜水抓龙虾时遭到一头鲨鱼的袭击并受伤。 | 官员说,在加利福尼亚州鱼季节开赛日,一条鱼袭击并伤害了一名13岁的男孩, |
| ntrex:0175:zho_Hans | news | What is certain is that the conflict of the mid 17th century has shaped the subsequent development of our nation, and Cromwell is an individual recognisable figure who represents one side of that divide. | 可以确定的是,17世纪中叶的冲突改变了我们国家后期的发展进程,而且克伦威尔是代表分歧一方的受到认可的人物。 | 肯定的是,17世纪中叶的冲突塑造了我们国家的后续发展, |
| ntrex:1320:zho_Hans | news | In her letter, Ms Freeman wrote: "Over the summer, negotiations between the UK and EU on withdrawal have continued, heading towards expected decisions this autumn. | 弗里曼在她的信中写道:“英国和欧盟之间关于脱欧的谈判持续了整个夏天,有望在今年秋季做出决定。 | 在她的信中,弗里曼女士写道:"在夏季期间,英国和欧盟之间的退出谈判继续进行, |

## Counterfactual sensitivity

| Pair | Category | Old fact in original | New fact absent in original | New fact appears | Old fact disappears | Unrelated chrF++ | All criteria |
|---|---|---:|---:|---:|---:|---:|---:|
| cf01_number | number | False | True | True | True | 43.84 | False |
| cf02_year | date | True | True | True | True | 100.00 | True |
| cf03_city | entity | True | True | True | True | 61.52 | False |
| cf04_calendar_date | date | True | True | True | True | 100.00 | True |
| cf05_decision | polarity | True | True | True | True | 100.00 | True |
| cf06_money | number | True | True | False | True | 39.09 | False |
| cf07_organization | entity | True | True | True | True | 55.26 | False |
| cf08_direction | polarity | True | True | True | True | 100.00 | True |
| cf09_weekday | date | True | True | True | True | 100.00 | True |
| cf10_patients | number | True | True | True | True | 100.00 | True |
