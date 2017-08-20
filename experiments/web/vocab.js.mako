
/*
Word count: ${len(tree['words'])}
*/

<%def name="node_r(node)" >\
    {
    collapsed: true,
    innerHTML: "<div class='vocab'><p>Code: ${node['codestr']}</p><div>${', '.join(w for w in node['shortwords'])}</div></div>",
    children: [
    % for c in node['children']:
        ${node_r(c)},
    % endfor\
    ]}
</%def>\

var chart_config = {
    chart: {
        container: "#vocab-tree",

        animateOnInit: true,

        node: {
            collapsable: true
        },
        animation: {
            nodeAnimation: "easeOutBounce",
            nodeSpeed: 700,
            connectorsAnimation: "bounce",
            connectorsSpeed: 700
        }
    },
    nodeStructure: {
                image: "figgs.png",
                collapsed: false,
        children: [
    % for c in tree['children']:
            ${node_r(c)},
    % endfor\
        ]
    }
};