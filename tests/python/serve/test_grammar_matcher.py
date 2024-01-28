# pylint: disable=missing-module-docstring,missing-function-docstring
import os

import pytest
import tvm
import tvm.testing
from tvm._ffi.base import TVMError

from mlc_chat.serve import BNFGrammar, GrammarMatcher


@pytest.fixture(scope="function")
def json_grammar():
    current_file_path = os.path.abspath(__file__)
    json_ebnf_path = os.path.join(os.path.dirname(current_file_path), "json.ebnf")

    with open(json_ebnf_path, "r", encoding="utf-8") as file:
        before = file.read()

    return BNFGrammar.from_ebnf_string(before)


# (json_inputs_accepted,) = tvm.testing.parameters(
#     ('{"name": "John"}',),
#     ('{"name": "Alice", "age": 30, "city": "New York"}',),
#     ('{"name": "Mike", "hobbies": ["reading", "cycling", "hiking"]}',),
#     ('{"name": "Emma", "address": {"street": "Maple Street", "city": "Boston"}}',),
#     ('[{"name": "David"}, {"name": "Sophia"}]',),
#     (
#         '{"name": "William", "age": null, "married": true, "children": ["Liam", "Olivia"],'
#         ' "hasPets": false}',
#     ),
#     (
#         '{"name": "Olivia", "contact": {"email": "olivia@example.com", "address": '
#         '{"city": "Chicago", "zipcode": "60601"}}}',
#     ),
#     (
#         '{"name": "Liam", "skills": ["Java", "Python"], "experience": '
#         '[{"company": "CompanyA", "years": 5}, {"company": "CompanyB", "years": 3}]}',
#     ),
#     (
#         '{"person": {"name": "Ethan", "age": 40}, "education": {"degree": "Masters", '
#         '"university": "XYZ University"}, "work": [{"company": "ABC Corp", "position": '
#         '"Manager"}, {"company": "DEF Corp", "position": "Senior Manager"}]}',
#     ),
#     (
#         '{"name": "Charlotte", "details": {"personal": {"age": 35, "hobbies": ["gardening", '
#         '"painting"]}, "professional": {"occupation": "Engineer", "skills": '
#         '["CAD", "Project Management"], "projects": [{"name": "Project A", '
#         '"status": "Completed"}, {"name": "Project B", "status": "In Progress"}]}}}',
#     ),
# )


# def test_json_accept(json_grammar: BNFGrammar, json_inputs_accepted: str):
#     print("grammar:", json_grammar)
#     matcher = GrammarMatcher(json_grammar)
#     assert matcher.can_match_complete_string(json_inputs_accepted)


# (json_inputs_refused,) = tvm.testing.parameters(
#     (r'{ name: "John" }',),
#     (r'{ "name": "John", "age": 30, }',),
#     (r'{ "name": "John", "address": { "street": "123 Main St", "city": "New York" }',),
#     (r'{ "name": "John", "age": 30, "hobbies": ["reading", "traveling",], }',),
#     (r'{ "name": "John", "age": 30.5.7 }',),
#     (r'{ "name": "John, "age": 30, "hobbies": ["reading", "traveling"] }',),
#     (
#         r'{ "name": "John", "age": 30, "hobbies": ["reading", { "type": "outdoor", "list": '
#         r'["hiking", "swimming",]}] }',
#     ),
#     (r'{ "name": "John", "age": 30, "status": "\P\J" }',),
#     (
#         r'{ "name": "John", "age": 30, "hobbies": ["reading", "traveling"], "address": '
#         r'{ "street": "123 Main St", "city": "New York", "coordinates": { "latitude": 40.7128, '
#         r'"longitude": -74.0060 }}}, "work": { "company": "Acme", "position": "developer" }}',
#     ),
# )


# def test_json_refuse(json_grammar: BNFGrammar, json_inputs_refused):
#     assert not GrammarMatcher(json_grammar).can_match_complete_string(json_inputs_refused)


(json_inputs_pressure,) = tvm.testing.parameters(
    # Extra long string: 1k chars
    # 0.1s on AMD Threadripper PRO 5975WX
    (
        '"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer nec odio. Praesent '
        "libero. Sed cursus ante dapibus diam. Sed nisi. Nulla quis sem at nibh elementum "
        "imperdiet. Duis sagittis ipsum. Praesent mauris. Fusce nec tellus sed augue semper "
        "porta. Mauris massa. Vestibulum lacinia arcu eget nulla. Class aptent taciti sociosqu "
        "ad litora torquent per conubia nostra, per inceptos himenaeos. Curabitur sodales ligula "
        "in libero. Sed dignissim lacinia nunc. Curabitur tortor. Pellentesque nibh. Aenean quam. "
        "In scelerisque sem at dolor. Maecenas mattis. Sed convallis tristique sem. Proin ut "
        "ligula vel nunc egestas porttitor. Morbi lectus risus, iaculis vel, suscipit quis, "
        "luctus non, massa. Fusce ac turpis quis ligula lacinia aliquet. Mauris ipsum. Nulla "
        "metus metus, ullamcorper vel, tincidunt sed, euismod in, nibh. Quisque volutpat "
        "condimentum velit. Class aptent taciti sociosqu ad litora torquent per conubia nostra, "
        "per inceptos himenaeos. Nam nec ante. Sed lacinia, urna non tincidunt mattis, tortor "
        "neque adipiscing diam, a cursus ipsum ante quis turpis. Nulla facilisi. Ut fringilla. "
        "Suspendisse potenti. Nunc feugiat mi a tellus consequat imperdiet. Vestibulum sapien. "
        "Proin quam. Etiam ultrices. Suspendisse in justo eu magna luctus suscipit. Sed lectus. "
        "Integer euismod lacus luctus magna. Quisque cursus, metus vitae pharetra auctor, sem "
        'massa mattis sem, at interdum magna augue eget diam."',
    ),
    # long and complex json: 3k chars
    # 30s on AMD Threadripper PRO 5975WX
    (
        r"""{"web-app": {
  "servlet": [
    {
      "servlet-name": "cofaxCDS",
      "servlet-class": "org.cofax.cds.CDSServlet",
      "init-param": {
        "configGlossary:installationAt": "Philadelphia, PA",
        "configGlossary:adminEmail": "ksm@pobox.com",
        "configGlossary:poweredBy": "Cofax",
        "configGlossary:poweredByIcon": "/images/cofax.gif",
        "configGlossary:staticPath": "/content/static",
        "templateProcessorClass": "org.cofax.WysiwygTemplate",
        "templateLoaderClass": "org.cofax.FilesTemplateLoader",
        "templatePath": "templates",
        "templateOverridePath": "",
        "defaultListTemplate": "listTemplate.htm",
        "defaultFileTemplate": "articleTemplate.htm",
        "useJSP": false,
        "jspListTemplate": "listTemplate.jsp",
        "jspFileTemplate": "articleTemplate.jsp",
        "cachePackageTagsTrack": 200,
        "cachePackageTagsStore": 200,
        "cachePackageTagsRefresh": 60,
        "cacheTemplatesTrack": 100,
        "cacheTemplatesStore": 50,
        "cacheTemplatesRefresh": 15,
        "cachePagesTrack": 200,
        "cachePagesStore": 100,
        "cachePagesRefresh": 10,
        "cachePagesDirtyRead": 10,
        "searchEngineListTemplate": "forSearchEnginesList.htm",
        "searchEngineFileTemplate": "forSearchEngines.htm",
        "searchEngineRobotsDb": "WEB-INF/robots.db",
        "useDataStore": true,
        "dataStoreClass": "org.cofax.SqlDataStore",
        "redirectionClass": "org.cofax.SqlRedirection",
        "dataStoreName": "cofax",
        "dataStoreDriver": "com.microsoft.jdbc.sqlserver.SQLServerDriver",
        "dataStoreUrl": "jdbc:microsoft:sqlserver://LOCALHOST:1433;DatabaseName=goon",
        "dataStoreUser": "sa",
        "dataStorePassword": "dataStoreTestQuery",
        "dataStoreTestQuery": "SET NOCOUNT ON;select test='test';",
        "dataStoreLogFile": "/usr/local/tomcat/logs/datastore.log",
        "dataStoreInitConns": 10,
        "dataStoreMaxConns": 100,
        "dataStoreConnUsageLimit": 100,
        "dataStoreLogLevel": "debug",
        "maxUrlLength": 500}},
    {
      "servlet-name": "cofaxEmail",
      "servlet-class": "org.cofax.cds.EmailServlet",
      "init-param": {
      "mailHost": "mail1",
      "mailHostOverride": "mail2"}},
    {
      "servlet-name": "cofaxAdmin",
      "servlet-class": "org.cofax.cds.AdminServlet"},

    {
      "servlet-name": "fileServlet",
      "servlet-class": "org.cofax.cds.FileServlet"},
    {
      "servlet-name": "cofaxTools",
      "servlet-class": "org.cofax.cms.CofaxToolsServlet",
      "init-param": {
        "templatePath": "toolstemplates/",
        "log": 1,
        "logLocation": "/usr/local/tomcat/logs/CofaxTools.log",
        "logMaxSize": "",
        "dataLog": 1,
        "dataLogLocation": "/usr/local/tomcat/logs/dataLog.log",
        "dataLogMaxSize": "",
        "removePageCache": "/content/admin/remove?cache=pages&id=",
        "removeTemplateCache": "/content/admin/remove?cache=templates&id=",
        "fileTransferFolder": "/usr/local/tomcat/webapps/content/fileTransferFolder",
        "lookInContext": 1,
        "adminGroupID": 4,
        "betaServer": true}}],
  "servlet-mapping": {
    "cofaxCDS": "/",
    "cofaxEmail": "/cofaxutil/aemail/*",
    "cofaxAdmin": "/admin/*",
    "fileServlet": "/static/*",
    "cofaxTools": "/tools/*"},
  "taglib": {
    "taglib-uri": "cofax.tld",
    "taglib-location": "/WEB-INF/tlds/cofax.tld"}}}
""",
    ),
)


def test_json_pressure(json_grammar: BNFGrammar, json_inputs_pressure):
    # 60s on AMD Threadripper 1950X. Need to be optimized later
    assert GrammarMatcher(json_grammar).can_match_complete_string(json_inputs_pressure)


if __name__ == "__main__":
    tvm.testing.main()
