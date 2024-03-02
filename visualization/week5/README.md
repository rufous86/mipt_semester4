<!doctype html>
<html>
<head>
<meta charset='UTF-8'><meta name='viewport' content='width=device-width initial-scale=1'>

<style type='text/css'>html {overflow-x: initial !important;}:root { --bg-color: #ffffff; --text-color: #333333; --select-text-bg-color: #B5D6FC; --select-text-font-color: auto; --monospace: "Lucida Console",Consolas,"Courier",monospace; --title-bar-height: 20px; }
.mac-os-11 { --title-bar-height: 28px; }
html { font-size: 14px; background-color: var(--bg-color); color: var(--text-color); font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; -webkit-font-smoothing: antialiased; }
h1, h2, h3, h4, h5 { white-space: pre-wrap; }
body { margin: 0px; padding: 0px; height: auto; inset: 0px; font-size: 1rem; line-height: 1.42857; overflow-x: hidden; background: inherit; }
iframe { margin: auto; }
a.url { word-break: break-all; }
a:active, a:hover { outline: 0px; }
.in-text-selection, ::selection { text-shadow: none; background: var(--select-text-bg-color); color: var(--select-text-font-color); }
#write { margin: 0px auto; height: auto; width: inherit; word-break: normal; overflow-wrap: break-word; position: relative; white-space: normal; overflow-x: visible; padding-top: 36px; }
#write.first-line-indent p { text-indent: 2em; }
#write.first-line-indent li p, #write.first-line-indent p * { text-indent: 0px; }
#write.first-line-indent li { margin-left: 2em; }
.for-image #write { padding-left: 8px; padding-right: 8px; }
body.typora-export { padding-left: 30px; padding-right: 30px; }
.typora-export .footnote-line, .typora-export li, .typora-export p { white-space: pre-wrap; }
.typora-export .task-list-item input { pointer-events: none; }
@media screen and (max-width: 500px) {
  body.typora-export { padding-left: 0px; padding-right: 0px; }
  #write { padding-left: 20px; padding-right: 20px; }
}
#write li > figure:last-child { margin-bottom: 0.5rem; }
#write ol, #write ul { position: relative; }
img { max-width: 100%; vertical-align: middle; image-orientation: from-image; }
button, input, select, textarea { color: inherit; font: inherit; }
input[type="checkbox"], input[type="radio"] { line-height: normal; padding: 0px; }
*, ::after, ::before { box-sizing: border-box; }
#write h1, #write h2, #write h3, #write h4, #write h5, #write h6, #write p, #write pre { width: inherit; }
#write h1, #write h2, #write h3, #write h4, #write h5, #write h6, #write p { position: relative; }
p { line-height: inherit; }
h1, h2, h3, h4, h5, h6 { break-after: avoid-page; break-inside: avoid; orphans: 4; }
p { orphans: 4; }
h1 { font-size: 2rem; }
h2 { font-size: 1.8rem; }
h3 { font-size: 1.6rem; }
h4 { font-size: 1.4rem; }
h5 { font-size: 1.2rem; }
h6 { font-size: 1rem; }
.md-math-block, .md-rawblock, h1, h2, h3, h4, h5, h6, p { margin-top: 1rem; margin-bottom: 1rem; }
.hidden { display: none; }
.md-blockmeta { color: rgb(204, 204, 204); font-weight: 700; font-style: italic; }
a { cursor: pointer; }
sup.md-footnote { padding: 2px 4px; background-color: rgba(238, 238, 238, 0.7); color: rgb(85, 85, 85); border-radius: 4px; cursor: pointer; }
sup.md-footnote a, sup.md-footnote a:hover { color: inherit; text-transform: inherit; text-decoration: inherit; }
#write input[type="checkbox"] { cursor: pointer; width: inherit; height: inherit; }
figure { overflow-x: auto; margin: 1.2em 0px; max-width: calc(100% + 16px); padding: 0px; }
figure > table { margin: 0px; }
thead, tr { break-inside: avoid; break-after: auto; }
thead { display: table-header-group; }
table { border-collapse: collapse; border-spacing: 0px; width: 100%; overflow: auto; break-inside: auto; text-align: left; }
table.md-table td { min-width: 32px; }
.CodeMirror-gutters { border-right: 0px; background-color: inherit; }
.CodeMirror-linenumber { user-select: none; }
.CodeMirror { text-align: left; }
.CodeMirror-placeholder { opacity: 0.3; }
.CodeMirror pre { padding: 0px 4px; }
.CodeMirror-lines { padding: 0px; }
div.hr:focus { cursor: none; }
#write pre { white-space: pre-wrap; }
#write.fences-no-line-wrapping pre { white-space: pre; }
#write pre.ty-contain-cm { white-space: normal; }
.CodeMirror-gutters { margin-right: 4px; }
.md-fences { font-size: 0.9rem; display: block; break-inside: avoid; text-align: left; overflow: visible; white-space: pre; background: inherit; position: relative !important; }
.md-fences-adv-panel { width: 100%; margin-top: 10px; text-align: center; padding-top: 0px; padding-bottom: 8px; overflow-x: auto; }
#write .md-fences.mock-cm { white-space: pre-wrap; }
.md-fences.md-fences-with-lineno { padding-left: 0px; }
#write.fences-no-line-wrapping .md-fences.mock-cm { white-space: pre; overflow-x: auto; }
.md-fences.mock-cm.md-fences-with-lineno { padding-left: 8px; }
.CodeMirror-line, twitterwidget { break-inside: avoid; }
svg { break-inside: avoid; }
.footnotes { opacity: 0.8; font-size: 0.9rem; margin-top: 1em; margin-bottom: 1em; }
.footnotes + .footnotes { margin-top: 0px; }
.md-reset { margin: 0px; padding: 0px; border: 0px; outline: 0px; vertical-align: top; background: 0px 0px; text-decoration: none; text-shadow: none; float: none; position: static; width: auto; height: auto; white-space: nowrap; cursor: inherit; -webkit-tap-highlight-color: transparent; line-height: normal; font-weight: 400; text-align: left; box-sizing: content-box; direction: ltr; }
li div { padding-top: 0px; }
blockquote { margin: 1rem 0px; }
li .mathjax-block, li p { margin: 0.5rem 0px; }
li blockquote { margin: 1rem 0px; }
li { margin: 0px; position: relative; }
blockquote > :last-child { margin-bottom: 0px; }
blockquote > :first-child, li > :first-child { margin-top: 0px; }
.footnotes-area { color: rgb(136, 136, 136); margin-top: 0.714rem; padding-bottom: 0.143rem; white-space: normal; }
#write .footnote-line { white-space: pre-wrap; }
@media print {
  body, html { border: 1px solid transparent; height: 99%; break-after: avoid; break-before: avoid; font-variant-ligatures: no-common-ligatures; }
  #write { margin-top: 0px; border-color: transparent !important; padding-top: 0px !important; padding-bottom: 0px !important; }
  .typora-export * { -webkit-print-color-adjust: exact; }
  .typora-export #write { break-after: avoid; }
  .typora-export #write::after { height: 0px; }
  .is-mac table { break-inside: avoid; }
  #write > p:nth-child(1) { margin-top: 0px; }
  .typora-export-show-outline .typora-export-sidebar { display: none; }
  figure { overflow-x: visible; }
}
.footnote-line { margin-top: 0.714em; font-size: 0.7em; }
a img, img a { cursor: pointer; }
pre.md-meta-block { font-size: 0.8rem; min-height: 0.8rem; white-space: pre-wrap; background: rgb(204, 204, 204); display: block; overflow-x: hidden; }
p > .md-image:only-child:not(.md-img-error) img, p > img:only-child { display: block; margin: auto; }
#write.first-line-indent p > .md-image:only-child:not(.md-img-error) img { left: -2em; position: relative; }
p > .md-image:only-child { display: inline-block; width: 100%; }
#write .MathJax_Display { margin: 0.8em 0px 0px; }
.md-math-block { width: 100%; }
.md-math-block:not(:empty)::after { display: none; }
.MathJax_ref { fill: currentcolor; }
[contenteditable="true"]:active, [contenteditable="true"]:focus, [contenteditable="false"]:active, [contenteditable="false"]:focus { outline: 0px; box-shadow: none; }
.md-task-list-item { position: relative; list-style-type: none; }
.task-list-item.md-task-list-item { padding-left: 0px; }
.md-task-list-item > input { position: absolute; top: 0px; left: 0px; margin-left: -1.2em; margin-top: calc(1em - 10px); border: none; }
.math { font-size: 1rem; }
.md-toc { min-height: 3.58rem; position: relative; font-size: 0.9rem; border-radius: 10px; }
.md-toc-content { position: relative; margin-left: 0px; }
.md-toc-content::after, .md-toc::after { display: none; }
.md-toc-item { display: block; color: rgb(65, 131, 196); }
.md-toc-item a { text-decoration: none; }
.md-toc-inner:hover { text-decoration: underline; }
.md-toc-inner { display: inline-block; cursor: pointer; }
.md-toc-h1 .md-toc-inner { margin-left: 0px; font-weight: 700; }
.md-toc-h2 .md-toc-inner { margin-left: 2em; }
.md-toc-h3 .md-toc-inner { margin-left: 4em; }
.md-toc-h4 .md-toc-inner { margin-left: 6em; }
.md-toc-h5 .md-toc-inner { margin-left: 8em; }
.md-toc-h6 .md-toc-inner { margin-left: 10em; }
@media screen and (max-width: 48em) {
  .md-toc-h3 .md-toc-inner { margin-left: 3.5em; }
  .md-toc-h4 .md-toc-inner { margin-left: 5em; }
  .md-toc-h5 .md-toc-inner { margin-left: 6.5em; }
  .md-toc-h6 .md-toc-inner { margin-left: 8em; }
}
a.md-toc-inner { font-size: inherit; font-style: inherit; font-weight: inherit; line-height: inherit; }
.footnote-line a:not(.reversefootnote) { color: inherit; }
.reversefootnote { font-family: ui-monospace, sans-serif; }
.md-attr { display: none; }
.md-fn-count::after { content: "."; }
code, pre, samp, tt { font-family: var(--monospace); }
kbd { margin: 0px 0.1em; padding: 0.1em 0.6em; font-size: 0.8em; color: rgb(36, 39, 41); background: rgb(255, 255, 255); border: 1px solid rgb(173, 179, 185); border-radius: 3px; box-shadow: rgba(12, 13, 14, 0.2) 0px 1px 0px, rgb(255, 255, 255) 0px 0px 0px 2px inset; white-space: nowrap; vertical-align: middle; }
.md-comment { color: rgb(162, 127, 3); opacity: 0.6; font-family: var(--monospace); }
code { text-align: left; vertical-align: initial; }
a.md-print-anchor { white-space: pre !important; border-width: initial !important; border-style: none !important; border-color: initial !important; display: inline-block !important; position: absolute !important; width: 1px !important; right: 0px !important; outline: 0px !important; background: 0px 0px !important; text-decoration: initial !important; text-shadow: initial !important; }
.os-windows.monocolor-emoji .md-emoji { font-family: "Segoe UI Symbol", sans-serif; }
.md-diagram-panel > svg { max-width: 100%; }
[lang="flow"] svg, [lang="mermaid"] svg { max-width: 100%; height: auto; }
[lang="mermaid"] .node text { font-size: 1rem; }
table tr th { border-bottom: 0px; }
video { max-width: 100%; display: block; margin: 0px auto; }
iframe { max-width: 100%; width: 100%; border: none; }
.highlight td, .highlight tr { border: 0px; }
mark { background: rgb(255, 255, 0); color: rgb(0, 0, 0); }
.md-html-inline .md-plain, .md-html-inline strong, mark .md-inline-math, mark strong { color: inherit; }
.md-expand mark .md-meta { opacity: 0.3 !important; }
mark .md-meta { color: rgb(0, 0, 0); }
@media print {
  .typora-export h1, .typora-export h2, .typora-export h3, .typora-export h4, .typora-export h5, .typora-export h6 { break-inside: avoid; }
}
.md-diagram-panel .messageText { stroke: none !important; }
.md-diagram-panel .start-state { fill: var(--node-fill); }
.md-diagram-panel .edgeLabel rect { opacity: 1 !important; }
.md-fences.md-fences-math { font-size: 1em; }
.md-fences-advanced:not(.md-focus) { padding: 0px; white-space: nowrap; border: 0px; }
.md-fences-advanced:not(.md-focus) { background: inherit; }
.typora-export-show-outline .typora-export-content { max-width: 1440px; margin: auto; display: flex; flex-direction: row; }
.typora-export-sidebar { width: 300px; font-size: 0.8rem; margin-top: 80px; margin-right: 18px; }
.typora-export-show-outline #write { --webkit-flex: 2; flex: 2 1 0%; }
.typora-export-sidebar .outline-content { position: fixed; top: 0px; max-height: 100%; overflow: hidden auto; padding-bottom: 30px; padding-top: 60px; width: 300px; }
@media screen and (max-width: 1024px) {
  .typora-export-sidebar, .typora-export-sidebar .outline-content { width: 240px; }
}
@media screen and (max-width: 800px) {
  .typora-export-sidebar { display: none; }
}
.outline-content li, .outline-content ul { margin-left: 0px; margin-right: 0px; padding-left: 0px; padding-right: 0px; list-style: none; overflow-wrap: anywhere; }
.outline-content ul { margin-top: 0px; margin-bottom: 0px; }
.outline-content strong { font-weight: 400; }
.outline-expander { width: 1rem; height: 1.42857rem; position: relative; display: table-cell; vertical-align: middle; cursor: pointer; padding-left: 4px; }
.outline-expander::before { content: ""; position: relative; font-family: Ionicons; display: inline-block; font-size: 8px; vertical-align: middle; }
.outline-item { padding-top: 3px; padding-bottom: 3px; cursor: pointer; }
.outline-expander:hover::before { content: ""; }
.outline-h1 > .outline-item { padding-left: 0px; }
.outline-h2 > .outline-item { padding-left: 1em; }
.outline-h3 > .outline-item { padding-left: 2em; }
.outline-h4 > .outline-item { padding-left: 3em; }
.outline-h5 > .outline-item { padding-left: 4em; }
.outline-h6 > .outline-item { padding-left: 5em; }
.outline-label { cursor: pointer; display: table-cell; vertical-align: middle; text-decoration: none; color: inherit; }
.outline-label:hover { text-decoration: underline; }
.outline-item:hover { border-color: rgb(245, 245, 245); background-color: var(--item-hover-bg-color); }
.outline-item:hover { margin-left: -28px; margin-right: -28px; border-left: 28px solid transparent; border-right: 28px solid transparent; }
.outline-item-single .outline-expander::before, .outline-item-single .outline-expander:hover::before { display: none; }
.outline-item-open > .outline-item > .outline-expander::before { content: ""; }
.outline-children { display: none; }
.info-panel-tab-wrapper { display: none; }
.outline-item-open > .outline-children { display: block; }
.typora-export .outline-item { padding-top: 1px; padding-bottom: 1px; }
.typora-export .outline-item:hover { margin-right: -8px; border-right: 8px solid transparent; }
.typora-export .outline-expander::before { content: "+"; font-family: inherit; top: -1px; }
.typora-export .outline-expander:hover::before, .typora-export .outline-item-open > .outline-item > .outline-expander::before { content: "−"; }
.typora-export-collapse-outline .outline-children { display: none; }
.typora-export-collapse-outline .outline-item-open > .outline-children, .typora-export-no-collapse-outline .outline-children { display: block; }
.typora-export-no-collapse-outline .outline-expander::before { content: "" !important; }
.typora-export-show-outline .outline-item-active > .outline-item .outline-label { font-weight: 700; }
.md-inline-math-container mjx-container { zoom: 0.95; }
mjx-container { break-inside: avoid; }
.md-alert.md-alert-note { border-left-color: rgb(9, 105, 218); }
.md-alert.md-alert-important { border-left-color: rgb(130, 80, 223); }
.md-alert.md-alert-warning { border-left-color: rgb(154, 103, 0); }
.md-alert.md-alert-tip { border-left-color: rgb(31, 136, 61); }
.md-alert.md-alert-caution { border-left-color: rgb(207, 34, 46); }
.md-alert { padding: 0px 1em; margin-bottom: 16px; color: inherit; border-left: 0.25em solid rgb(0, 0, 0); }
.md-alert-text-note { color: rgb(9, 105, 218); }
.md-alert-text-important { color: rgb(130, 80, 223); }
.md-alert-text-warning { color: rgb(154, 103, 0); }
.md-alert-text-tip { color: rgb(31, 136, 61); }
.md-alert-text-caution { color: rgb(207, 34, 46); }
.md-alert-text { font-size: 0.9rem; font-weight: 700; }
.md-alert-text svg { fill: currentcolor; position: relative; top: 0.125em; margin-right: 1ch; overflow: visible; }
.md-alert-text-container::after { content: attr(data-text); text-transform: capitalize; pointer-events: none; margin-right: 1ch; }


.CodeMirror { height: auto; }
.CodeMirror.cm-s-inner { background: inherit; }
.CodeMirror-scroll { overflow: auto hidden; z-index: 3; }
.CodeMirror-gutter-filler, .CodeMirror-scrollbar-filler { background-color: rgb(255, 255, 255); }
.CodeMirror-gutters { border-right: 1px solid rgb(221, 221, 221); background: inherit; white-space: nowrap; }
.CodeMirror-linenumber { padding: 0px 3px 0px 5px; text-align: right; color: rgb(153, 153, 153); }
.cm-s-inner .cm-keyword { color: rgb(119, 0, 136); }
.cm-s-inner .cm-atom, .cm-s-inner.cm-atom { color: rgb(34, 17, 153); }
.cm-s-inner .cm-number { color: rgb(17, 102, 68); }
.cm-s-inner .cm-def { color: rgb(0, 0, 255); }
.cm-s-inner .cm-variable { color: rgb(0, 0, 0); }
.cm-s-inner .cm-variable-2 { color: rgb(0, 85, 170); }
.cm-s-inner .cm-variable-3 { color: rgb(0, 136, 85); }
.cm-s-inner .cm-string { color: rgb(170, 17, 17); }
.cm-s-inner .cm-property { color: rgb(0, 0, 0); }
.cm-s-inner .cm-operator { color: rgb(152, 26, 26); }
.cm-s-inner .cm-comment, .cm-s-inner.cm-comment { color: rgb(170, 85, 0); }
.cm-s-inner .cm-string-2 { color: rgb(255, 85, 0); }
.cm-s-inner .cm-meta { color: rgb(85, 85, 85); }
.cm-s-inner .cm-qualifier { color: rgb(85, 85, 85); }
.cm-s-inner .cm-builtin { color: rgb(51, 0, 170); }
.cm-s-inner .cm-bracket { color: rgb(153, 153, 119); }
.cm-s-inner .cm-tag { color: rgb(17, 119, 0); }
.cm-s-inner .cm-attribute { color: rgb(0, 0, 204); }
.cm-s-inner .cm-header, .cm-s-inner.cm-header { color: rgb(0, 0, 255); }
.cm-s-inner .cm-quote, .cm-s-inner.cm-quote { color: rgb(0, 153, 0); }
.cm-s-inner .cm-hr, .cm-s-inner.cm-hr { color: rgb(153, 153, 153); }
.cm-s-inner .cm-link, .cm-s-inner.cm-link { color: rgb(0, 0, 204); }
.cm-negative { color: rgb(221, 68, 68); }
.cm-positive { color: rgb(34, 153, 34); }
.cm-header, .cm-strong { font-weight: 700; }
.cm-del { text-decoration: line-through; }
.cm-em { font-style: italic; }
.cm-link { text-decoration: underline; }
.cm-error { color: red; }
.cm-invalidchar { color: red; }
.cm-constant { color: rgb(38, 139, 210); }
.cm-defined { color: rgb(181, 137, 0); }
div.CodeMirror span.CodeMirror-matchingbracket { color: rgb(0, 255, 0); }
div.CodeMirror span.CodeMirror-nonmatchingbracket { color: rgb(255, 34, 34); }
.cm-s-inner .CodeMirror-activeline-background { background: inherit; }
.CodeMirror { position: relative; overflow: hidden; }
.CodeMirror-scroll { height: 100%; outline: 0px; position: relative; box-sizing: content-box; background: inherit; }
.CodeMirror-sizer { position: relative; }
.CodeMirror-gutter-filler, .CodeMirror-hscrollbar, .CodeMirror-scrollbar-filler, .CodeMirror-vscrollbar { position: absolute; z-index: 6; display: none; outline: 0px; }
.CodeMirror-vscrollbar { right: 0px; top: 0px; overflow: hidden; }
.CodeMirror-hscrollbar { bottom: 0px; left: 0px; overflow: auto hidden; }
.CodeMirror-scrollbar-filler { right: 0px; bottom: 0px; }
.CodeMirror-gutter-filler { left: 0px; bottom: 0px; }
.CodeMirror-gutters { position: absolute; left: 0px; top: 0px; padding-bottom: 10px; z-index: 3; overflow-y: hidden; }
.CodeMirror-gutter { white-space: normal; height: 100%; box-sizing: content-box; padding-bottom: 30px; margin-bottom: -32px; display: inline-block; }
.CodeMirror-gutter-wrapper { position: absolute; z-index: 4; background: 0px 0px !important; border: none !important; }
.CodeMirror-gutter-background { position: absolute; top: 0px; bottom: 0px; z-index: 4; }
.CodeMirror-gutter-elt { position: absolute; cursor: default; z-index: 4; }
.CodeMirror-lines { cursor: text; }
.CodeMirror pre { border-radius: 0px; border-width: 0px; background: 0px 0px; font-family: inherit; font-size: inherit; margin: 0px; white-space: pre; overflow-wrap: normal; color: inherit; z-index: 2; position: relative; overflow: visible; }
.CodeMirror-wrap pre { overflow-wrap: break-word; white-space: pre-wrap; word-break: normal; }
.CodeMirror-code pre { border-right: 30px solid transparent; width: fit-content; }
.CodeMirror-wrap .CodeMirror-code pre { border-right: none; width: auto; }
.CodeMirror-linebackground { position: absolute; inset: 0px; z-index: 0; }
.CodeMirror-linewidget { position: relative; z-index: 2; overflow: auto; }
.CodeMirror-wrap .CodeMirror-scroll { overflow-x: hidden; }
.CodeMirror-measure { position: absolute; width: 100%; height: 0px; overflow: hidden; visibility: hidden; }
.CodeMirror-measure pre { position: static; }
.CodeMirror div.CodeMirror-cursor { position: absolute; visibility: hidden; border-right: none; width: 0px; }
.CodeMirror div.CodeMirror-cursor { visibility: hidden; }
.CodeMirror-focused div.CodeMirror-cursor { visibility: inherit; }
.cm-searching { background: rgba(255, 255, 0, 0.4); }
span.cm-underlined { text-decoration: underline; }
span.cm-strikethrough { text-decoration: line-through; }
.cm-tw-syntaxerror { color: rgb(255, 255, 255); background-color: rgb(153, 0, 0); }
.cm-tw-deleted { text-decoration: line-through; }
.cm-tw-header5 { font-weight: 700; }
.cm-tw-listitem:first-child { padding-left: 10px; }
.cm-tw-box { border-style: solid; border-right-width: 1px; border-bottom-width: 1px; border-left-width: 1px; border-color: inherit; border-top-width: 0px !important; }
.cm-tw-underline { text-decoration: underline; }
@media print {
  .CodeMirror div.CodeMirror-cursor { visibility: hidden; }
}


:root {
  --mermaid-theme: night;
}

[lang='mermaid'] .label {
  color: #333;
}

/* CSS Document */

/** code highlight */

.cm-s-inner .cm-variable,
.cm-s-inner .cm-operator,
.cm-s-inner .cm-property {
    color: #b8bfc6;
}

.cm-s-inner .cm-keyword {
    color: #C88FD0;
}

.cm-s-inner .cm-tag {
    color: #7DF46A;
}

.cm-s-inner .cm-attribute {
    color: #7575E4;
}

.CodeMirror div.CodeMirror-cursor {
    border-left: 1px solid #b8bfc6;
    z-index: 3;
}

.cm-s-inner .cm-string {
    color: #D26B6B;
}

.cm-s-inner .cm-comment,
.cm-s-inner.cm-comment {
    color: #DA924A;
}

.cm-s-inner .cm-header,
.cm-s-inner .cm-def,
.cm-s-inner.cm-header,
.cm-s-inner.cm-def {
    color: #8d8df0;
}

.cm-s-inner .cm-quote,
.cm-s-inner.cm-quote {
    color: #57ac57;
}

.cm-s-inner .cm-hr {
    color: #d8d5d5;
}

.cm-s-inner .cm-link {
    color: #d3d3ef;
}

.cm-s-inner .cm-negative {
    color: #d95050;
}

.cm-s-inner .cm-positive {
    color: #50e650;
}

.cm-s-inner .cm-string-2 {
    color: #f50;
}

.cm-s-inner .cm-meta,
.cm-s-inner .cm-qualifier {
    color: #b7b3b3;
}

.cm-s-inner .cm-builtin {
    color: #f3b3f8;
}

.cm-s-inner .cm-bracket {
    color: #997;
}

.cm-s-inner .cm-atom,
.cm-s-inner.cm-atom {
    color: #84B6CB;
}

.cm-s-inner .cm-number {
    color: #64AB8F;
}

.cm-s-inner .cm-variable {
    color: #b8bfc6;
}

.cm-s-inner .cm-variable-2 {
    color: #9FBAD5;
}

.cm-s-inner .cm-variable-3 {
    color: #1cc685;
}

.CodeMirror-selectedtext,
.CodeMirror-selected {
    background: #4a89dc;
    color: #fff !important;
    text-shadow: none;
}

.CodeMirror-gutters {
    border-right: none;
}

/* CSS Document */

/** markdown source **/
.cm-s-typora-default .cm-header, 
.cm-s-typora-default .cm-property
{
    color: #cebcca;
}

.CodeMirror.cm-s-typora-default div.CodeMirror-cursor{
    border-left: 3px solid #b8bfc6;
}

.cm-s-typora-default .cm-comment {
    color: #9FB1FF;
}

.cm-s-typora-default .cm-string {
    color: #A7A7D9
}

.cm-s-typora-default .cm-atom, .cm-s-typora-default .cm-number {
    color: #848695;
    font-style: italic;
}

.cm-s-typora-default .cm-link {
    color: #95B94B;
}

.cm-s-typora-default .CodeMirror-activeline-background {
    background: rgba(51, 51, 51, 0.72);
}

.cm-s-typora-default .cm-comment, .cm-s-typora-default .cm-code {
	color: #8aa1e1;
}@import "";
@import "";
@import "";

:root {
    --bg-color:  #363B40;
    --side-bar-bg-color: #2E3033;
    --text-color: #b8bfc6;

    --select-text-bg-color:#4a89dc;

    --item-hover-bg-color: #0a0d16;
    --control-text-color: #b7b7b7;
    --control-text-hover-color: #eee;
    --window-border: 1px solid #555;

    --active-file-bg-color: rgb(34, 34, 34);
    --active-file-border-color: #8d8df0;

    --primary-color: #a3d5fe;

    --active-file-text-color: white;
    --item-hover-bg-color: #70717d;
    --item-hover-text-color: white;
    --primary-color: #6dc1e7;

    --rawblock-edit-panel-bd: #333;

    --search-select-bg-color: #428bca;
}

html {
    font-size: 16px;
    -webkit-font-smoothing: antialiased;
}

html,
body {
    -webkit-text-size-adjust: 100%;
    -ms-text-size-adjust: 100%;
    background: #363B40;
    background: var(--bg-color);
    fill: currentColor;
    line-height: 1.625rem;
}

#write {
    max-width: 914px;
}


@media only screen and (min-width: 1400px) {
	#write {
		max-width: 1024px;
	}
}

@media only screen and (min-width: 1800px) {
	#write {
		max-width: 1200px;
	}
}

html,
body,
button,
input,
select,
textarea,
div.code-tooltip-content {
    color: #b8bfc6;
    border-color: transparent;
}

div.code-tooltip,
.md-hover-tip .md-arrow:after {
    background: #333;
}

.native-window #md-notification {
    border: 1px solid #70717d;
}

.popover.bottom > .arrow:after {
    border-bottom-color: #333;
}

html,
body,
button,
input,
select,
textarea {
    font-family: "Helvetica Neue", Helvetica, Arial, 'Segoe UI Emoji', sans-serif;
}

hr {
    height: 2px;
    border: 0;
    margin: 24px 0 !important;
}

h1,
h2,
h3,
h4,
h5,
h6 {
    font-family: "Lucida Grande", "Corbel", sans-serif;
    font-weight: normal;
    clear: both;
    -ms-word-wrap: break-word;
    word-wrap: break-word;
    margin: 0;
    padding: 0;
    color: #DEDEDE
}

h1 {
    font-size: 2.5rem;
    /* 36px */
    line-height: 2.75rem;
    /* 40px */
    margin-bottom: 1.5rem;
    /* 24px */
    letter-spacing: -1.5px;
}

h2 {
    font-size: 1.63rem;
    /* 24px */
    line-height: 1.875rem;
    /* 30px */
    margin-bottom: 1.5rem;
    /* 24px */
    letter-spacing: -1px;
    font-weight: bold;
}

h3 {
    font-size: 1.17rem;
    /* 18px */
    line-height: 1.5rem;
    /* 24px */
    margin-bottom: 1.5rem;
    /* 24px */
    letter-spacing: -1px;
    font-weight: bold;
}

h4 {
    font-size: 1.12rem;
    /* 16px */
    line-height: 1.375rem;
    /* 22px */
    margin-bottom: 1.5rem;
    /* 24px */
    color: white;
}

h5 {
    font-size: 0.97rem;
    /* 16px */
    line-height: 1.25rem;
    /* 22px */
    margin-bottom: 1.5rem;
    /* 24px */
    font-weight: bold;
}

h6 {
    font-size: 0.93rem;
    /* 16px */
    line-height: 1rem;
    /* 16px */
    margin-bottom: 0.75rem;
    color: white;
}

@media (min-width: 980px) {
    h3.md-focus:before,
    h4.md-focus:before,
    h5.md-focus:before,
    h6.md-focus:before {
        color: #ddd;
        border: 1px solid #ddd;
        border-radius: 3px;
        position: absolute;
        left: -1.642857143rem;
        top: .357142857rem;
        float: left;
        font-size: 9px;
        padding-left: 2px;
        padding-right: 2px;
        vertical-align: bottom;
        font-weight: normal;
        line-height: normal;
    }

    h3.md-focus:before {
        content: 'h3';
    }

    h4.md-focus:before {
        content: 'h4';
    }

    h5.md-focus:before {
        content: 'h5';
        top: 0px;
    }

    h6.md-focus:before {
        content: 'h6';
        top: 0px;
    }
}

a {
    text-decoration: none;
    outline: 0;
}

a:hover {
    outline: 0;
}

a:focus {
    outline: thin dotted;
}

sup.md-footnote {
    background-color: #555;
    color: #ddd;
}

p {
    -ms-word-wrap: break-word;
    word-wrap: break-word;
}

p,
ul,
dd,
ol,
hr,
address,
pre,
table,
iframe,
.wp-caption,
.wp-audio-shortcode,
.wp-video-shortcode {
    margin-top: 0;
    margin-bottom: 1.5rem;
    /* 24px */
}

audio:not([controls]) {
    display: none;
}

[hidden] {
    display: none;
}

::-moz-selection {
    background: #4a89dc;
    color: #fff;
    text-shadow: none;
}

*.in-text-selection,
::selection {
    background: #4a89dc;
    color: #fff;
    text-shadow: none;
}

ul,
ol {
    padding: 0 0 0 1.875rem;
    /* 30px */
}

ul {
    list-style: square;
}

ol {
    list-style: decimal;
}

ul ul,
ol ol,
ul ol,
ol ul {
    margin: 0;
}

b,
th,
dt,
strong {
    font-weight: bold;
}

i,
em,
dfn,
cite {
    font-style: italic;
}

blockquote {
    padding-left: 1.875rem;
    margin: 0 0 1.875rem 1.875rem;
    border-left: solid 2px #474d54;
    padding-left: 30px;
    margin-top: 35px;
}

pre,
code,
kbd,
tt,
var {
    font-size: 0.875em;
    font-family: Monaco, Consolas, "Andale Mono", "DejaVu Sans Mono", monospace;
}

code,
tt,
var {
    background: rgba(0, 0, 0, 0.05);
}

kbd {
    padding: 2px 4px;
    font-size: 90%;
    color: #fff;
    background-color: #333;
    border-radius: 3px;
    box-shadow: inset 0 -1px 0 rgba(0,0,0,.25);
}

pre.md-fences {
    padding: 10px 10px 10px 30px;
    margin-bottom: 20px;
    background: #333;
}

.CodeMirror-gutters {
    background: #333;
    border-right: 1px solid transparent;
}

.enable-diagrams pre.md-fences[lang="sequence"] .code-tooltip,
.enable-diagrams pre.md-fences[lang="flow"] .code-tooltip,
.enable-diagrams pre.md-fences[lang="mermaid"] .code-tooltip {
    bottom: -2.2em;
    right: 4px;
}

code,
kbd,
tt,
var {
    padding: 2px 5px;
}

table {
    max-width: 100%;
    width: 100%;
    border-collapse: collapse;
    border-spacing: 0;
}

th,
td {
    padding: 5px 10px;
    vertical-align: top;
}

a {
    -webkit-transition: all .2s ease-in-out;
    transition: all .2s ease-in-out;
}

hr {
    background: #474d54;
    /* variable */
}

h1 {
    margin-top: 2em;
}

a {
    color: #e0e0e0;
    text-decoration: underline;
}

a:hover {
    color: #fff;
}

.md-inline-math script {
    color: #81b1db;
}

b,
th,
dt,
strong {
    color: #DEDEDE;
    /* variable */
}

mark {
    background: #D3D40E;
}

blockquote {
    color: #9DA2A6;
}

table a {
    color: #DEDEDE;
    /* variable */
}

th,
td {
    border: solid 1px #474d54;
    /* variable */
}

.task-list {
    padding-left: 0;
}

.md-task-list-item {
    padding-left: 1.25rem;
}

.md-task-list-item > input {
    top: auto;
}

.md-task-list-item > input:before {
    content: "";
    display: inline-block;
    width: 0.875rem;
    height: 0.875rem;
    vertical-align: middle;
    text-align: center;
    border: 1px solid #b8bfc6;
    background-color: #363B40;
    margin-top: -0.4rem;
}

.md-task-list-item > input:checked:before,
.md-task-list-item > input[checked]:before {
    content: '\221A';
    /*◘*/
    font-size: 0.625rem;
    line-height: 0.625rem;
    color: #DEDEDE;
}

/** quick open **/
.auto-suggest-container {
    border: 0px;
    background-color: #525C65;
}

#typora-quick-open {
    background-color: #525C65;
}

#typora-quick-open input{
    background-color: #525C65;
    border: 0;
    border-bottom: 1px solid grey;
}

.typora-quick-open-item {
    background-color: inherit;
    color: inherit;
}

.typora-quick-open-item.active,
.typora-quick-open-item:hover {
    background-color: #4D8BDB;
    color: white;
}

.typora-quick-open-item:hover {
    background-color: rgba(77, 139, 219, 0.8);
}

.typora-search-spinner > div {
  background-color: #fff;
}

#write pre.md-meta-block {
    border-bottom: 1px dashed #ccc;
    background: transparent;
    padding-bottom: 0.6em;
    line-height: 1.6em;
}

.btn,
.btn .btn-default {
    background: transparent;
    color: #b8bfc6;
}

.ty-table-edit {
    border-top: 1px solid gray;
    background-color: #363B40;
}

.popover-title {
    background: transparent;
}

.md-image>.md-meta {
    color: #BBBBBB;
    background: transparent;
}

.md-expand.md-image>.md-meta {
    color: #DDD;
}

#write>h3:before,
#write>h4:before,
#write>h5:before,
#write>h6:before {
    border: none;
    border-radius: 0px;
    color: #888;
    text-decoration: underline;
    left: -1.4rem;
    top: 0.2rem;
}

#write>h3.md-focus:before {
    top: 2px;
}

#write>h4.md-focus:before {
    top: 2px;
}

.md-toc-item {
    color: #A8C2DC;
}

#write div.md-toc-tooltip {
    background-color: #363B40;
}

.dropdown-menu .btn:hover,
.dropdown-menu .btn:focus,
.md-toc .btn:hover,
.md-toc .btn:focus {
    color: white;
    background: black;
}

#toc-dropmenu {
    background: rgba(50, 54, 59, 0.93);
    border: 1px solid rgba(253, 253, 253, 0.15);
}

#toc-dropmenu .divider {
    background-color: #9b9b9b;
}

.outline-expander:before {
    top: 2px;
}

#typora-sidebar {
    box-shadow: none;
    border-right: 1px dashed;
    border-right: none;
}

.sidebar-tabs {
    border-bottom:0;
}

#typora-sidebar:hover .outline-title-wrapper {
    border-left: 1px dashed;
}

.outline-title-wrapper .btn {
    color: inherit;
}

.outline-item:hover {
    border-color: #363B40;
    background-color: #363B40;
    color: white;
}

h1.md-focus .md-attr,
h2.md-focus .md-attr,
h3.md-focus .md-attr,
h4.md-focus .md-attr,
h5.md-focus .md-attr,
h6.md-focus .md-attr,
.md-header-span .md-attr {
    color: #8C8E92;
    display: inline;
}

.md-comment {
    color: #5a95e3;
    opacity: 0.8;
}

.md-inline-math svg {
    color: #b8bfc6;
}

#math-inline-preview .md-arrow:after {
    background: black;
}

.modal-content {
    background: var(--bg-color);
    border: 0;
}

.modal-title {
    font-size: 1.5em;
}

.modal-content input {
    background-color: rgba(26, 21, 21, 0.51);
    color: white;
}

.modal-content .input-group-addon {
    color: white;
}

.modal-backdrop {
    background-color: rgba(174, 174, 174, 0.7);
}

.modal-content .btn-primary {
    border-color: var(--primary-color);
}

.md-table-resize-popover {
    background-color: #333;
}

.form-inline .input-group .input-group-addon {
    color: white;
}

#md-searchpanel {
    border-bottom: 1px dashed grey;
}

/** UI for electron */

.context-menu,
#spell-check-panel,
#footer-word-count-info {
    background-color: #42464A;
}

.context-menu.dropdown-menu .divider,
.dropdown-menu .divider {
    background-color: #777777;
    opacity: 1;
}

footer {
    color: inherit;
}

@media (max-width: 1000px) {
    footer {
        border-top: none;
    }
    footer:hover {
        color: inherit;
    }
}

#file-info-file-path .file-info-field-value:hover {
    background-color: #555;
    color: #dedede;
}

.megamenu-content,
.megamenu-opened header {
    background: var(--bg-color);
}

.megamenu-menu-panel h2,
.megamenu-menu-panel h1,
.long-btn {
    color: inherit;
}

.megamenu-menu-panel input[type='text'] {
    background: inherit;
    border: 0;
    border-bottom: 1px solid;
}

#recent-file-panel-action-btn {
    background: inherit;
    border: 1px grey solid;
}

.megamenu-menu-panel .dropdown-menu > li > a {
    color: inherit;
    background-color: #2F353A;
    text-decoration: none;
}

.megamenu-menu-panel table td:nth-child(1) {
    color: inherit;
    font-weight: bold;
}

.megamenu-menu-panel tbody tr:hover td:nth-child(1) {
    color: white;
}

.modal-footer .btn-default, 
.modal-footer .btn-primary,
.modal-footer .btn-default:not(:hover) {
    border: 1px solid;
    border-color: transparent;
}

.btn-primary {
    color: white;
}

.btn-default:hover, .btn-default:focus, .btn-default.focus, .btn-default:active, .btn-default.active, .open > .dropdown-toggle.btn-default {
    color: white;
    border: 1px solid #ddd;
    background-color: inherit;
}

.modal-header {
    border-bottom: 0;
}

.modal-footer {
    border-top: 0;
}

#recent-file-panel tbody tr:nth-child(2n-1) {
    background-color: transparent !important;
}

.megamenu-menu-panel tbody tr:hover td:nth-child(2) {
    color: inherit;
}

.megamenu-menu-panel .btn {
    border: 1px solid #eee;
    background: transparent;
}

.mouse-hover .toolbar-icon.btn:hover,
#w-full.mouse-hover,
#w-pin.mouse-hover {
    background-color: inherit;
}

.typora-node::-webkit-scrollbar {
    width: 5px;
}

.typora-node::-webkit-scrollbar-thumb:vertical {
    background: rgba(250, 250, 250, 0.3);
}

.typora-node::-webkit-scrollbar-thumb:vertical:active {
    background: rgba(250, 250, 250, 0.5);
}

#w-unpin {
    background-color: #4182c4;
}

#top-titlebar, #top-titlebar * {
    color: var(--item-hover-text-color);
}

.typora-sourceview-on #toggle-sourceview-btn,
#footer-word-count:hover,
.ty-show-word-count #footer-word-count {
    background: #333333;
}

#toggle-sourceview-btn:hover {
    color: #eee;
    background: #333333;
}

/** focus mode */
.on-focus-mode .md-end-block:not(.md-focus):not(.md-focus-container) * {
    color: #686868 !important;
}

.on-focus-mode .md-end-block:not(.md-focus) img,
.on-focus-mode .md-task-list-item:not(.md-focus-container)>input {
    opacity: #686868 !important;
}

.on-focus-mode li[cid]:not(.md-focus-container){
    color: #686868;
}

.on-focus-mode .md-fences.md-focus .CodeMirror-code>*:not(.CodeMirror-activeline) *,
.on-focus-mode .CodeMirror.cm-s-inner:not(.CodeMirror-focused) * {
    color: #686868 !important;
}

.on-focus-mode .md-focus,
.on-focus-mode .md-focus-container {
    color: #fff;
}

.on-focus-mode #typora-source .CodeMirror-code>*:not(.CodeMirror-activeline) * {
    color: #686868 !important;
}


/*diagrams*/
#write .md-focus .md-diagram-panel {
    border: 1px solid #ddd;
    margin-left: -1px;
    width: calc(100% + 2px);
}

/*diagrams*/
#write .md-focus.md-fences-with-lineno .md-diagram-panel {
    margin-left: auto;
}

.md-diagram-panel-error {
    color: #f1908e;
}

.active-tab-files #info-panel-tab-file,
.active-tab-files #info-panel-tab-file:hover,
.active-tab-outline #info-panel-tab-outline,
.active-tab-outline #info-panel-tab-outline:hover {
    color: #eee;
}

.sidebar-footer-item:hover,
.footer-item:hover {
    background: inherit;
    color: white;
}

.ty-side-sort-btn.active,
.ty-side-sort-btn:hover,
.selected-folder-menu-item a:after {
    color: white;
}

#sidebar-files-menu {
    border:solid 1px;
    box-shadow: 4px 4px 20px rgba(0, 0, 0, 0.79);
    background-color: var(--bg-color);
}

.file-list-item {
    border-bottom:none;
}

.file-list-item-summary {
    opacity: 1;
}

.file-list-item.active:first-child {
    border-top: none;
}

.file-node-background {
    height: 32px;
}

.file-library-node.active>.file-node-content,
.file-list-item.active {
    color: white;
    color: var(--active-file-text-color);
}

.file-library-node.active>.file-node-background{
    background-color: rgb(34, 34, 34);
    background-color: var(--active-file-bg-color);
}
.file-list-item.active {
    background-color: rgb(34, 34, 34);
    background-color: var(--active-file-bg-color);
}

#ty-tooltip {
    background-color: black;
    color: #eee;
}

.md-task-list-item>input {
    margin-left: -1.3em;
    margin-top: 0.3rem;
    -webkit-appearance: none;
}

.md-mathjax-midline {
    background-color: #57616b;
    border-bottom: none;
}

footer.ty-footer {
    border-color: #656565;
}

.ty-preferences .btn-default {
    background: transparent;
}
.ty-preferences .btn-default:hover {
    background: #57616b;
}

.ty-preferences select {
    border: 1px solid #989698;
    height: 21px;
}

.ty-preferences .nav-group-item.active,
.export-item.active,
.export-items-list-control,
.export-detail {
    background: var(--item-hover-bg-color);
}

.ty-preferences input[type="search"] {
    border-color: #333;
    background: #333;
    line-height: 22px;
    border-radius: 6px;
    color: white;
}

.ty-preferences input[type="search"]:focus {
    box-shadow: none;
}

[data-is-directory="true"] .file-node-content {
    margin-bottom: 0;
}

.file-node-title {
    line-height: 22px;
}

.html-for-mac .file-node-open-state, .html-for-mac .file-node-icon {
    line-height: 26px;
}

::-webkit-scrollbar-thumb {
    background: rgba(230, 230, 230, 0.30);
}

::-webkit-scrollbar-thumb:active {
    background: rgba(230, 230, 230, 0.50);
}

#typora-sidebar:hover div.sidebar-content-content::-webkit-scrollbar-thumb:horizontal {
    background: rgba(230, 230, 230, 0.30);
}

.nav-group-item:active {
    background-color: #474d54 !important;
}

.md-search-hit {
    background: rgba(199, 140, 60, 0.81);
    color: #eee;
}

.md-search-hit * {
    color: #eee;
}

#md-searchpanel input {
    color: white;
}

.modal-backdrop.in {
    opacity: 1;
    backdrop-filter: blur(1px);
}

.clear-btn-icon {
    top: 8px;
}

/* try fix https://github.com/typora/typora-issues/issues/5253 */
.file-node-expanded>.file-node-children {
    display: grid;
  }

.md-alert-text-note {
    color: rgb(47, 129, 247);
}
.md-alert-text-important {
    color: rgb(163, 113, 247);
}
.md-alert-text-warning {
    color:  rgb(210, 153, 34);
}


</style><title>5-dimension-reduction</title>
</head>
<body class='typora-export'><div class='typora-export-content'>
<div id='write'  class=''><h1 id='методы-понижения-размерности-и-оценка-результатов-работы-алгоритмов'><span>Методы понижения размерности и оценка результатов работы алгоритмов</span></h1><blockquote><p><strong><span>Алгоритмы понижения размерности</span></strong><span> — это методы анализа данных, которые используются для уменьшения количества переменных или признаков в наборе данных, стараясь сохранить как можно больше информации о данных. Они играют важную роль в анализе данных и машинном обучении: позволяют уменьшить избыточность данных, выделить наиболее значимые признаки и улучшить производительность алгоритмов машинного обучения.</span></p></blockquote><p><span>Распространенные алгоритмы понижения размерности:</span></p><ul><li><p><strong><span>Метод главных компонент (PCA).</span></strong><span> PCA находит линейные комбинации исходных признаков, которые максимально сохраняют дисперсию данных. Он проецирует данные на новое пространство меньшей размерности, называемое главными компонентами.</span></p></li><li><p><strong><span>t-распределенное стохастическое вложение соседей (t-SNE).</span></strong><span> t-SNE пытается сохранить локальные структуры данных, то есть близость точек в исходном пространстве в новом пространстве меньшей размерности. Он часто используется для визуализации данных в двумерном или трехмерном пространстве.</span></p></li><li><p><strong><span>Линейный дискриминантный анализ (LDA).</span></strong><span> LDA ищет линейные комбинации признаков, которые максимизируют различия между классами в данных. Он часто используется в задачах классификации.</span></p></li><li><p><strong><span>Нелинейные методы понижения размерности.</span></strong><span> К ним относятся такие методы, как Isomap, Locally Linear Embedding (LLE), Autoencoders. Они захватывают более сложные нелинейные зависимости в данных.</span></p></li><li><p><strong><span>Многомерное шкалирование (MDS).</span></strong><span> MDS стремится сохранить расстояния между точками в исходном пространстве в новом пространстве меньшей размерности. Это позволяет визуализировать исходные данные в пространстве меньшей размерности.</span></p></li></ul><p><span>Сегодня мы рассмотрим самые главные и самые распространенные из этих методов. Начнем работу с импорта библиотек:</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python" style="break-inside: unset;"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><span><span>​</span>x</span></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation" style=""><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">gzip</span></span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">pandas</span> <span class="cm-keyword">as</span> <span class="cm-variable">pd</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">numpy</span> <span class="cm-keyword">as</span> <span class="cm-variable">np</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">plotly</span>.<span class="cm-property">offline</span> <span class="cm-keyword">as</span> <span class="cm-variable">py</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">py</span>.<span class="cm-property">init_notebook_mode</span>(<span class="cm-variable">connected</span><span class="cm-operator">=</span><span class="cm-keyword">True</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">plotly</span>.<span class="cm-property">graph_objs</span> <span class="cm-keyword">as</span> <span class="cm-variable">go</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">plotly</span>.<span class="cm-property">tools</span> <span class="cm-keyword">as</span> <span class="cm-variable">tls</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">seaborn</span> <span class="cm-keyword">as</span> <span class="cm-variable">sns</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">matplotlib</span>.<span class="cm-property">image</span> <span class="cm-keyword">as</span> <span class="cm-variable">mpimg</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">matplotlib</span>.<span class="cm-property">pyplot</span> <span class="cm-keyword">as</span> <span class="cm-variable">plt</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">matplotlib</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">time</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-operator">%</span><span class="cm-variable">matplotlib</span> <span class="cm-variable">inline</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Import the 3 dimensionality reduction methods</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">manifold</span> <span class="cm-keyword">import</span> <span class="cm-variable">TSNE</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">decomposition</span> <span class="cm-keyword">import</span> <span class="cm-variable">PCA</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">discriminant_analysis</span> <span class="cm-keyword">import</span> <span class="cm-variable">LinearDiscriminantAnalysis</span> <span class="cm-keyword">as</span> <span class="cm-variable">LDA</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 572px;"></div><div class="CodeMirror-gutters" style="display: none; height: 572px;"></div></div></div></pre><p><span>Мы будем работать с популярным датасетом </span><a href='https://en.wikipedia.org/wiki/MNIST_database'><span>MNIST</span></a><span>. Этот датасет представляет собой изображения картинок размером 28 на 28 пикселей. На каждой картинке изображены цифры от нуля до девяти. В рамках нашего датасета каждый пиксель содержится в отдельном столбце. Итого 785 столбцов — один отвечает за label (какая цифра нарисована на самом деле), остальные — за цвет в конкретном пикселе.</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation"><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">train</span> <span class="cm-operator">=</span> <span class="cm-variable">pd</span>.<span class="cm-property">read_csv</span>(<span class="cm-string">'https://github.com/ElijahSum/mipt_visualization/raw/master/week_05_visualization/mnist_train.csv'</span>, <span class="cm-variable">compression</span><span class="cm-operator">=</span><span class="cm-string">'gzip'</span>, <span class="cm-variable">index_col</span><span class="cm-operator">=</span><span class="cm-number">0</span>)</span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">train</span>.<span class="cm-property">head</span>(<span class="cm-number">10</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 130px;"></div><div class="CodeMirror-gutters" style="display: none; height: 130px;"></div></div></div></pre><figure class='table-figure'><table><thead><tr><th><strong><span>label</span></strong></th><th><strong><span>1x1</span></strong></th><th><strong><span>1x2</span></strong></th><th><strong><span>1x3</span></strong></th><th><strong><span>1x4</span></strong></th><th><strong><span>1x5</span></strong></th><th><strong><span>1x6</span></strong></th><th><strong><span>1x7</span></strong></th><th><strong><span>1x8</span></strong></th><th><strong><span>1x9</span></strong></th><th><strong><span>...</span></strong></th><th><strong><span>28x19</span></strong></th><th><strong><span>28x20</span></strong></th><th><strong><span>28x21</span></strong></th><th><strong><span>28x22</span></strong></th><th><strong><span>28x23</span></strong></th><th><strong><span>28x24</span></strong></th><th><strong><span>28x25</span></strong></th><th><strong><span>28x26</span></strong></th><th><strong><span>28x27</span></strong></th><th><strong><span>28x28</span></strong></th><th>&nbsp;</th></tr></thead><tbody><tr><td><strong><span>0</span></strong></td><td><span>5</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>...</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td></tr><tr><td><strong><span>1</span></strong></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>...</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td></tr><tr><td><strong><span>2</span></strong></td><td><span>4</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>...</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td></tr><tr><td><strong><span>3</span></strong></td><td><span>1</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>...</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td></tr><tr><td><strong><span>4</span></strong></td><td><span>9</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>...</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td></tr><tr><td><strong><span>5</span></strong></td><td><span>2</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>...</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td></tr><tr><td><strong><span>6</span></strong></td><td><span>1</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>...</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td></tr><tr><td><strong><span>7</span></strong></td><td><span>3</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>...</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td></tr><tr><td><strong><span>8</span></strong></td><td><span>1</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>...</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td></tr><tr><td><strong><span>9</span></strong></td><td><span>4</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>...</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td><td><span>0</span></td></tr></tbody></table></figure><p><span>Теперь разделим наш датасет на две части — </span><code>train</code><span> и </span><code>test</code><span>, чтобы использовать их в работе с алгоритмами.</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python" style="break-inside: unset;"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation" style=""><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">target</span> <span class="cm-operator">=</span> <span class="cm-variable">train</span>[<span class="cm-string">'label'</span>]</span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">train</span> <span class="cm-operator">=</span> <span class="cm-variable">train</span>.<span class="cm-property">drop</span>(<span class="cm-string">"label"</span>,<span class="cm-variable">axis</span><span class="cm-operator">=</span><span class="cm-number">1</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">target</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-number">0</span> &nbsp; &nbsp; &nbsp; &nbsp;<span class="cm-number">5</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-number">1</span> &nbsp; &nbsp; &nbsp; &nbsp;<span class="cm-number">0</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-number">2</span> &nbsp; &nbsp; &nbsp; &nbsp;<span class="cm-number">4</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-number">3</span> &nbsp; &nbsp; &nbsp; &nbsp;<span class="cm-number">1</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-number">4</span> &nbsp; &nbsp; &nbsp; &nbsp;<span class="cm-number">9</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"> &nbsp; &nbsp; &nbsp;  ..</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-number">59995</span> &nbsp; &nbsp;<span class="cm-number">8</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-number">59996</span> &nbsp; &nbsp;<span class="cm-number">3</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-number">59997</span> &nbsp; &nbsp;<span class="cm-number">5</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-number">59998</span> &nbsp; &nbsp;<span class="cm-number">6</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-number">59999</span> &nbsp; &nbsp;<span class="cm-number">8</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">Name</span>: <span class="cm-variable">label</span>, <span class="cm-variable">Length</span>: <span class="cm-number">60000</span>, <span class="cm-variable">dtype</span>: <span class="cm-variable">int64</span></span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 416px;"></div><div class="CodeMirror-gutters" style="display: none; height: 416px;"></div></div></div></pre><p><span>Отрисуем одну из цифр нашего датасета. Для этого воспользуемся </span><code>plt.imshow()</code><span>. Эта функция позволяет попиксельно вывести изображение.</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation"><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">imshow</span>(<span class="cm-variable">np</span>.<span class="cm-property">array</span>([<span class="cm-number">1</span>, <span class="cm-number">2</span>, <span class="cm-number">3</span>, <span class="cm-number">2</span>, <span class="cm-number">4</span>, <span class="cm-number">2</span>, <span class="cm-number">3</span>, <span class="cm-number">2</span>, <span class="cm-number">1</span>]).<span class="cm-property">reshape</span>(<span class="cm-number">3</span>,<span class="cm-number">3</span>))</span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">axis</span>(<span class="cm-string">'off'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;">(<span class="cm-operator">-</span><span class="cm-number">0.5</span>, <span class="cm-number">2.5</span>, <span class="cm-number">2.5</span>, <span class="cm-operator">-</span><span class="cm-number">0.5</span>)</span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 78px;"></div><div class="CodeMirror-gutters" style="display: none; height: 78px;"></div></div></div></pre><p><img src="/home/andrei/study/mipt_semester4/visualization/week5/assets/image-20240302101910083.png" referrerpolicy="no-referrer" alt="image-20240302101910083"></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation"><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">figure</span>(<span class="cm-variable">figsize</span><span class="cm-operator">=</span>(<span class="cm-number">5</span>, <span class="cm-number">3</span>))</span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">imshow</span>(<span class="cm-variable">train</span>.<span class="cm-property">iloc</span>[<span class="cm-number">10000</span>].<span class="cm-property">to_numpy</span>().<span class="cm-property">reshape</span>(<span class="cm-number">28</span>,<span class="cm-number">28</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">axis</span>(<span class="cm-string">'off'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">show</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 104px;"></div><div class="CodeMirror-gutters" style="display: none; height: 104px;"></div></div></div></pre><p><img src="/home/andrei/study/mipt_semester4/visualization/week5/assets/image-20240302101946198.png" referrerpolicy="no-referrer" alt="image-20240302101946198"></p><p><span>Теперь отрисуем 70 чисел из нашего датасета, чтобы посмотреть, насколько сильно они отличаются.</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation" style=""><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">figure</span>(<span class="cm-variable">figsize</span><span class="cm-operator">=</span>(<span class="cm-number">10</span>, <span class="cm-number">6</span>))</span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">for</span> <span class="cm-variable">digit_num</span> <span class="cm-keyword">in</span> <span class="cm-builtin">range</span>(<span class="cm-number">0</span>,<span class="cm-number">70</span>):</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">subplot</span>(<span class="cm-number">7</span>,<span class="cm-number">10</span>,<span class="cm-variable">digit_num</span><span class="cm-operator">+</span><span class="cm-number">1</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">grid_data</span> <span class="cm-operator">=</span> <span class="cm-variable">train</span>.<span class="cm-property">iloc</span>[<span class="cm-variable">digit_num</span>].<span class="cm-property">to_numpy</span>().<span class="cm-property">reshape</span>(<span class="cm-number">28</span>,<span class="cm-number">28</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">imshow</span>(<span class="cm-variable">grid_data</span>, <span class="cm-variable">interpolation</span> <span class="cm-operator">=</span> <span class="cm-string">"none"</span>, <span class="cm-variable">cmap</span> <span class="cm-operator">=</span> <span class="cm-string">"afmhot"</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">xticks</span>([])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">yticks</span>([])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">tight_layout</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 208px;"></div><div class="CodeMirror-gutters" style="display: none; height: 208px;"></div></div></div></pre><p><img src="/home/andrei/study/mipt_semester4/visualization/week5/assets/image-20240302102027509.png" referrerpolicy="no-referrer" alt="image-20240302102027509"></p><h3 id='зачем-использовать-алгоритмы-понижения-размерности'><strong><span>Зачем использовать алгоритмы понижения размерности:</span></strong></h3><ul><li><p><span>Уменьшение размерности для повышения производительности, чтобы алгоритм работал быстрее и не обращал внимания на неинформативные признаки.</span></p></li><li><p><span>Уменьшение размерности для построения визуализаций. Преобразование в двумерное пространство позволит построить привычную визуализацию, которая отразит зависимости в данных.</span></p></li></ul><p><span>Алгоритмы снижения размерности позволяют, свести N-мерный массив данных (датасет) к двумерному, сохраняя внутренние связи между данными. Это позволяет строить понятные глазу визуализации, которые можно интерпретировать.</span></p><p><span>Каждую из этих целей мы рассмотрим отдельно в рамках текущего занятия. Начнем обзор с самого популярного и простого метода — PCA.</span></p><h2 id='principal-component-analysis-pca'><span>Principal Component Analysis (PCA)</span></h2><blockquote><p><strong><span>Principal Component Analysis (PCA)</span></strong><span> — это метод понижения размерности и одновременно метод анализа данных, который используется для идентификации основных шаблонов в данных. Он находит новые оси, называемые главными компонентами, которые представляют собой линейные комбинации исходных признаков. Главные компоненты упорядочены по убыванию дисперсии данных. Это означает, что первая главная компонента объясняет наибольшую долю общей дисперсии, вторая — следующую по величине долю и т. д.</span></p></blockquote><p><span>В библиотеке </span><code>sklearn</code><span> уже есть реализация алгоритма PCA. Перед началом нашей работы отскалируем данные с помощью </span><code>StandardScaler()</code><span>:</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation" style=""><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">нашей</span> <span class="cm-variable">работы</span> <span class="cm-variable">отскалируем</span> <span class="cm-variable">данные</span> <span class="cm-variable">с</span> <span class="cm-variable">помощью</span> <span class="cm-variable">StandardScaler</span>():</span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Standardize the data</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">preprocessing</span> <span class="cm-keyword">import</span> <span class="cm-variable">StandardScaler</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X</span> <span class="cm-operator">=</span> <span class="cm-variable">train</span>.<span class="cm-property">values</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X_std</span> <span class="cm-operator">=</span> <span class="cm-variable">StandardScaler</span>().<span class="cm-property">fit_transform</span>(<span class="cm-variable">X</span>)</span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 130px;"></div><div class="CodeMirror-gutters" style="display: none; height: 130px;"></div></div></div></pre><p><span>PCA принимает на вход как минимум один параметр — </span><code>n_components</code><span>.</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation"><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Invoke SKlearn's PCA method</span></span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">n_components</span> <span class="cm-operator">=</span> <span class="cm-number">95</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">pca</span> <span class="cm-operator">=</span> <span class="cm-variable">PCA</span>(<span class="cm-variable">n_components</span><span class="cm-operator">=</span><span class="cm-variable">n_components</span>).<span class="cm-property">fit</span>(<span class="cm-variable">train</span>.<span class="cm-property">values</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X_reduced</span> <span class="cm-operator">=</span> <span class="cm-variable">pca</span>.<span class="cm-property">fit_transform</span>(<span class="cm-variable">train</span>)</span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 104px;"></div><div class="CodeMirror-gutters" style="display: none; height: 104px;"></div></div></div></pre><p><span>Мы применили простейший метод понижения размерности. Преимущество этих методов в том, что они обратимы, — после мы можем восстановить наши значения, например, с помощью функции pca.inverse_transform().</span></p><p><span>Посмотрим, сильно ли отличается восстановленное изображение от оригинала:</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation" style=""><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment">#now that we have got both compressed and decompressed data, let's plot them side by side to see how much we lost(5%)</span></span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">fig</span>, (<span class="cm-variable">ax1</span>, <span class="cm-variable">ax2</span>) <span class="cm-operator">=</span> <span class="cm-variable">plt</span>.<span class="cm-property">subplots</span>(<span class="cm-number">1</span>,<span class="cm-number">2</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X_decompress</span> <span class="cm-operator">=</span> <span class="cm-variable">pca</span>.<span class="cm-property">inverse_transform</span>(<span class="cm-variable">X_reduced</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">ax1</span>.<span class="cm-property">imshow</span>(<span class="cm-variable">train</span>.<span class="cm-property">iloc</span>[<span class="cm-number">10000</span>].<span class="cm-property">to_numpy</span>().<span class="cm-property">reshape</span>(<span class="cm-number">28</span>,<span class="cm-number">28</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">ax2</span>.<span class="cm-property">imshow</span>(<span class="cm-variable">X_decompress</span>[<span class="cm-number">10000</span>].<span class="cm-property">reshape</span>(<span class="cm-number">28</span>, <span class="cm-number">28</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">fig</span>.<span class="cm-property">suptitle</span>(<span class="cm-string">'compression and decompression'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">ax1</span>.<span class="cm-property">axis</span>(<span class="cm-string">'off'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">ax2</span>.<span class="cm-property">axis</span>(<span class="cm-string">'off'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">show</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 260px;"></div><div class="CodeMirror-gutters" style="display: none; height: 260px;"></div></div></div></pre><p><img src="/home/andrei/study/mipt_semester4/visualization/week5/assets/image-20240302102153259.png" referrerpolicy="no-referrer" alt="image-20240302102153259"></p><p><span>Мы видим, что полученное новое значение не сильно отличается от исходного. Да, видны переходы между цветами, но изображение все еще различимо и принципиально не отличается.</span></p><p><span>Методы понижения размерности могут применяться для повышения производительности при работе с алгоритмами. Попробуем применить разные классификаторы к исходному датасету с 784 признаками, а затем к датасету после применения алгоритма понижения размерности. После этого сравним их производительность.</span></p><h2 id='random-forest-classifier'><span>Random Forest Classifier</span></h2><p><span>Для сверки времени работы воспользуемся разницей в </span><code>time.time()</code><span>. Применим алгоритм на исходном датасете:</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python" style="break-inside: unset;"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation" style=""><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">алгоритм</span> <span class="cm-variable">на</span> <span class="cm-variable">исходном</span> <span class="cm-variable">датасете</span>:</span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">ensemble</span> <span class="cm-keyword">import</span> <span class="cm-variable">RandomForestClassifier</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">rfc</span> <span class="cm-operator">=</span> <span class="cm-variable">RandomForestClassifier</span>(<span class="cm-variable">n_estimators</span><span class="cm-operator">=</span><span class="cm-number">20</span>, <span class="cm-variable">random_state</span> <span class="cm-operator">=</span> <span class="cm-number">42</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">t0</span> <span class="cm-operator">=</span> <span class="cm-variable">time</span>.<span class="cm-property">time</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">rfc</span>.<span class="cm-property">fit</span>(<span class="cm-variable">train</span>, <span class="cm-variable">target</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">t1</span> <span class="cm-operator">=</span> <span class="cm-variable">time</span>.<span class="cm-property">time</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">rfc</span>.<span class="cm-property">score</span>(<span class="cm-variable">train</span>, <span class="cm-variable">target</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">t1</span><span class="cm-operator">-</span><span class="cm-variable">t0</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-number">0.9999333333333333</span> <span class="cm-number">14.608989953994751</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">rfc2</span> <span class="cm-operator">=</span> <span class="cm-variable">RandomForestClassifier</span>(<span class="cm-variable">n_estimators</span> <span class="cm-operator">=</span> <span class="cm-number">20</span>, <span class="cm-variable">random_state</span> <span class="cm-operator">=</span> <span class="cm-number">42</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">t0</span> <span class="cm-operator">=</span> <span class="cm-variable">time</span>.<span class="cm-property">time</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">rfc2</span>.<span class="cm-property">fit</span>(<span class="cm-variable">X_reduced</span>, <span class="cm-variable">target</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">t1</span> <span class="cm-operator">=</span> <span class="cm-variable">time</span>.<span class="cm-property">time</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">rfc2</span>.<span class="cm-property">score</span>(<span class="cm-variable">X_reduced</span>, <span class="cm-variable">target</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">t1</span><span class="cm-operator">-</span><span class="cm-variable">t0</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-number">0.9999666666666667</span> <span class="cm-number">41.03299927711487</span></span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 442px;"></div><div class="CodeMirror-gutters" style="display: none; height: 442px;"></div></div></div></pre><p><span>Что-то пошло не так — скорость вычисления не только не уменьшилась, но даже возросла в три раза. Зато незначительно выросло качество классификации. Посмотрим, как это будет работать на другом алгоритме.</span></p><p><span>В домашнем задании мы разберем эту ситуацию подробнее и построим визуализацию работы дерева.</span></p><h2 id='softmax-classifier'><span>Softmax Classifier</span></h2><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python" style="break-inside: unset;"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation" style=""><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">linear_model</span> <span class="cm-keyword">import</span> <span class="cm-variable">LogisticRegression</span></span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">log_clf</span> <span class="cm-operator">=</span> <span class="cm-variable">LogisticRegression</span>(<span class="cm-variable">random_state</span> <span class="cm-operator">=</span> <span class="cm-number">42</span>, <span class="cm-variable">max_iter</span><span class="cm-operator">=</span><span class="cm-number">200</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">t0</span> <span class="cm-operator">=</span> <span class="cm-variable">time</span>.<span class="cm-property">time</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">log_clf</span>.<span class="cm-property">fit</span>(<span class="cm-variable">train</span>, <span class="cm-variable">target</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">t1</span> <span class="cm-operator">=</span> <span class="cm-variable">time</span>.<span class="cm-property">time</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">log_clf</span>.<span class="cm-property">score</span>(<span class="cm-variable">train</span>, <span class="cm-variable">target</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">t1</span><span class="cm-operator">-</span><span class="cm-variable">t0</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">linear_model</span> <span class="cm-keyword">import</span> <span class="cm-variable">LogisticRegression</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">log_clf</span> <span class="cm-operator">=</span> <span class="cm-variable">LogisticRegression</span>(<span class="cm-variable">multi_class</span> <span class="cm-operator">=</span> <span class="cm-string">'multinomial'</span>, <span class="cm-variable">solver</span> <span class="cm-operator">=</span> <span class="cm-string">'lbfgs'</span>, <span class="cm-variable">random_state</span> <span class="cm-operator">=</span> <span class="cm-number">42</span>, <span class="cm-variable">max_iter</span><span class="cm-operator">=</span><span class="cm-number">200</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">t0</span> <span class="cm-operator">=</span> <span class="cm-variable">time</span>.<span class="cm-property">time</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">log_clf</span>.<span class="cm-property">fit</span>(<span class="cm-variable">train</span>, <span class="cm-variable">target</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">t1</span> <span class="cm-operator">=</span> <span class="cm-variable">time</span>.<span class="cm-property">time</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">log_clf</span>.<span class="cm-property">score</span>(<span class="cm-variable">train</span>, <span class="cm-variable">target</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">t1</span><span class="cm-operator">-</span><span class="cm-variable">t0</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-operator">/</span><span class="cm-variable">usr</span><span class="cm-operator">/</span><span class="cm-variable">local</span><span class="cm-operator">/</span><span class="cm-variable">lib</span><span class="cm-operator">/</span><span class="cm-variable">python3</span><span class="cm-number">.10</span><span class="cm-operator">/</span><span class="cm-variable">dist</span><span class="cm-operator">-</span><span class="cm-variable">packages</span><span class="cm-operator">/</span><span class="cm-variable">sklearn</span><span class="cm-operator">/</span><span class="cm-variable">linear_model</span><span class="cm-operator">/</span><span class="cm-variable">_logistic</span>.<span class="cm-property">py</span>:<span class="cm-number">458</span>: <span class="cm-variable">ConvergenceWarning</span>:</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">lbfgs</span> <span class="cm-variable">failed</span> <span class="cm-variable">to</span> <span class="cm-variable">converge</span> (<span class="cm-variable">status</span><span class="cm-operator">=</span><span class="cm-number">1</span>):</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">STOP</span>: <span class="cm-variable">TOTAL</span> <span class="cm-variable">NO</span>. <span class="cm-property">of</span> <span class="cm-variable">ITERATIONS</span> <span class="cm-variable">REACHED</span> <span class="cm-variable">LIMIT</span>.</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-property">Increase</span> <span class="cm-variable">the</span> <span class="cm-variable">number</span> <span class="cm-variable">of</span> <span class="cm-variable">iterations</span> (<span class="cm-variable">max_iter</span>) <span class="cm-keyword">or</span> <span class="cm-variable">scale</span> <span class="cm-variable">the</span> <span class="cm-variable">data</span> <span class="cm-keyword">as</span> <span class="cm-variable">shown</span> <span class="cm-keyword">in</span>:</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"> &nbsp; &nbsp;<span class="cm-variable">https</span>:<span class="cm-operator">//</span><span class="cm-variable">scikit</span><span class="cm-operator">-</span><span class="cm-variable">learn</span>.<span class="cm-property">org</span><span class="cm-operator">/</span><span class="cm-variable">stable</span><span class="cm-operator">/</span><span class="cm-variable">modules</span><span class="cm-operator">/</span><span class="cm-variable">preprocessing</span>.<span class="cm-property">html</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">Please</span> <span class="cm-variable">also</span> <span class="cm-variable">refer</span> <span class="cm-variable">to</span> <span class="cm-variable">the</span> <span class="cm-variable">documentation</span> <span class="cm-keyword">for</span> <span class="cm-variable">alternative</span> <span class="cm-variable">solver</span> <span class="cm-variable">options</span>:</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"> &nbsp; &nbsp;<span class="cm-variable">https</span>:<span class="cm-operator">//</span><span class="cm-variable">scikit</span><span class="cm-operator">-</span><span class="cm-variable">learn</span>.<span class="cm-property">org</span><span class="cm-operator">/</span><span class="cm-variable">stable</span><span class="cm-operator">/</span><span class="cm-variable">modules</span><span class="cm-operator">/</span><span class="cm-variable">linear_model</span>.<span class="cm-property">html</span><span class="cm-comment">#logistic-regression</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-number">0.9379</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-number">96.47242832183838</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">log_clf2</span> <span class="cm-operator">=</span> <span class="cm-variable">LogisticRegression</span>(<span class="cm-variable">multi_class</span> <span class="cm-operator">=</span> <span class="cm-string">'multinomial'</span>, <span class="cm-variable">solver</span> <span class="cm-operator">=</span> <span class="cm-string">'lbfgs'</span>, <span class="cm-variable">random_state</span> <span class="cm-operator">=</span> <span class="cm-number">42</span>, <span class="cm-variable">max_iter</span><span class="cm-operator">=</span><span class="cm-number">500</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">t0</span> <span class="cm-operator">=</span> <span class="cm-variable">time</span>.<span class="cm-property">time</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">log_clf2</span>.<span class="cm-property">fit</span>(<span class="cm-variable">X_reduced</span>, <span class="cm-variable">target</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">t1</span> <span class="cm-operator">=</span> <span class="cm-variable">time</span>.<span class="cm-property">time</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">log_clf2</span>.<span class="cm-property">score</span>(<span class="cm-variable">X_reduced</span>, <span class="cm-variable">target</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">t1</span><span class="cm-operator">-</span><span class="cm-variable">t0</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-number">0.92115</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-number">34.53950333595276</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-operator">/</span><span class="cm-variable">usr</span><span class="cm-operator">/</span><span class="cm-variable">local</span><span class="cm-operator">/</span><span class="cm-variable">lib</span><span class="cm-operator">/</span><span class="cm-variable">python3</span><span class="cm-number">.10</span><span class="cm-operator">/</span><span class="cm-variable">dist</span><span class="cm-operator">-</span><span class="cm-variable">packages</span><span class="cm-operator">/</span><span class="cm-variable">sklearn</span><span class="cm-operator">/</span><span class="cm-variable">linear_model</span><span class="cm-operator">/</span><span class="cm-variable">_logistic</span>.<span class="cm-property">py</span>:<span class="cm-number">458</span>: <span class="cm-variable">ConvergenceWarning</span>:</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">lbfgs</span> <span class="cm-variable">failed</span> <span class="cm-variable">to</span> <span class="cm-variable">converge</span> (<span class="cm-variable">status</span><span class="cm-operator">=</span><span class="cm-number">1</span>):</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">STOP</span>: <span class="cm-variable">TOTAL</span> <span class="cm-variable">NO</span>. <span class="cm-property">of</span> <span class="cm-variable">ITERATIONS</span> <span class="cm-variable">REACHED</span> <span class="cm-variable">LIMIT</span>.</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-property">Increase</span> <span class="cm-variable">the</span> <span class="cm-variable">number</span> <span class="cm-variable">of</span> <span class="cm-variable">iterations</span> (<span class="cm-variable">max_iter</span>) <span class="cm-keyword">or</span> <span class="cm-variable">scale</span> <span class="cm-variable">the</span> <span class="cm-variable">data</span> <span class="cm-keyword">as</span> <span class="cm-variable">shown</span> <span class="cm-keyword">in</span>:</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"> &nbsp; &nbsp;<span class="cm-variable">https</span>:<span class="cm-operator">//</span><span class="cm-variable">scikit</span><span class="cm-operator">-</span><span class="cm-variable">learn</span>.<span class="cm-property">org</span><span class="cm-operator">/</span><span class="cm-variable">stable</span><span class="cm-operator">/</span><span class="cm-variable">modules</span><span class="cm-operator">/</span><span class="cm-variable">preprocessing</span>.<span class="cm-property">html</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">Please</span> <span class="cm-variable">also</span> <span class="cm-variable">refer</span> <span class="cm-variable">to</span> <span class="cm-variable">the</span> <span class="cm-variable">documentation</span> <span class="cm-keyword">for</span> <span class="cm-variable">alternative</span> <span class="cm-variable">solver</span> <span class="cm-variable">options</span>:</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"> &nbsp; &nbsp;<span class="cm-variable">https</span>:<span class="cm-operator">//</span><span class="cm-variable">scikit</span><span class="cm-operator">-</span><span class="cm-variable">learn</span>.<span class="cm-property">org</span><span class="cm-operator">/</span><span class="cm-variable">stable</span><span class="cm-operator">/</span><span class="cm-variable">modules</span><span class="cm-operator">/</span><span class="cm-variable">linear_model</span>.<span class="cm-property">html</span><span class="cm-comment">#logistic-regression</span></span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 1430px;"></div><div class="CodeMirror-gutters" style="display: none; height: 1430px;"></div></div></div></pre><p><span>Теперь мы видим совершенно другую ситуацию — мы смогли ускорить работу алгоритма аж в три раза, не сильно потеряв в итоговом качестве.</span></p><p><span>Таким образом, при работе с рядом алгоритмов сокращение размерности позволяет увеличить скорость работы без значительных потерь в качестве.</span></p><h1 id='графическое-представление-данных-пониженной-размерности'><span>Графическое представление данных пониженной размерности</span></h1><p><span>Другое применение алгоритмов понижения размерности — возможность создавать визуализации, чтобы лучше понять происходящее в данных (например, насколько близко друг к другу находятся разные классы). Проблемой может быть то, что датасет — это n-мерный массив данных, который плохо подходит, чтобы строить двумерную визуализацию. Чтобы отразить состояние данных, их нужно свести к двумерному датасету.</span></p><p><span>Главная польза работы таких алгоритмов — не нужно выделять оси, по которым строится визуализация. Наша работа сводится к тому, чтобы запустить алгоритм и посмотреть, есть ли в нем какие-либо визуальные паттерны.</span></p><h2 id='pca'><span>PCA</span></h2><p><span>Еще раз запустим алгоритм PCA, но теперь для построения быстрой и эффективной визуализации оставим только 10% от всех наших наблюдений. После снижения размерности данных запустим на нем несколько алгоритмов кластеризации и посмотрим, получится ли найти на них зависимости.</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python" style="break-inside: unset;"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation" style=""><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X</span><span class="cm-operator">=</span> <span class="cm-variable">train</span>[:<span class="cm-number">6000</span>].<span class="cm-property">values</span></span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X_std</span> <span class="cm-operator">=</span> <span class="cm-variable">StandardScaler</span>().<span class="cm-property">fit_transform</span>(<span class="cm-variable">X</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">pca</span> <span class="cm-operator">=</span> <span class="cm-variable">PCA</span>(<span class="cm-variable">n_components</span><span class="cm-operator">=</span><span class="cm-number">5</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">pca</span>.<span class="cm-property">fit</span>(<span class="cm-variable">X_std</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X_5d</span> <span class="cm-operator">=</span> <span class="cm-variable">pca</span>.<span class="cm-property">transform</span>(<span class="cm-variable">X_std</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">Target</span> <span class="cm-operator">=</span> <span class="cm-variable">target</span>[:<span class="cm-number">6000</span>]</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">matplotlib</span>.<span class="cm-property">pyplot</span> <span class="cm-keyword">as</span> <span class="cm-variable">plt</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">seaborn</span> <span class="cm-keyword">as</span> <span class="cm-variable">sns</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Create the scatter plot using matplotlib</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">figure</span>(<span class="cm-variable">figsize</span><span class="cm-operator">=</span>(<span class="cm-number">10</span>, <span class="cm-number">8</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">sns</span>.<span class="cm-property">scatterplot</span>(</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">x</span><span class="cm-operator">=</span><span class="cm-variable">X_5d</span>[:, <span class="cm-number">0</span>],</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">y</span><span class="cm-operator">=</span><span class="cm-variable">X_5d</span>[:, <span class="cm-number">1</span>],</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">hue</span><span class="cm-operator">=</span><span class="cm-variable">Target</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">palette</span><span class="cm-operator">=</span><span class="cm-string">'Accent'</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">legend</span><span class="cm-operator">=</span><span class="cm-string">'full'</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">edgecolor</span><span class="cm-operator">=</span><span class="cm-string">'white'</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">linewidth</span><span class="cm-operator">=</span><span class="cm-number">0.5</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">alpha</span><span class="cm-operator">=</span><span class="cm-number">0.8</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;">)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Set the title and labels</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">title</span>(<span class="cm-string">'Principal Component Analysis (PCA)'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">xlabel</span>(<span class="cm-string">'First Principal Component'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">ylabel</span>(<span class="cm-string">'Second Principal Component'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Show the legend</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">legend</span>(<span class="cm-variable">title</span><span class="cm-operator">=</span><span class="cm-string">'Target'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Show the plot</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">grid</span>(<span class="cm-keyword">True</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">show</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 1248px;"></div><div class="CodeMirror-gutters" style="display: none; height: 1248px;"></div></div></div></pre><p><img src="/home/andrei/study/mipt_semester4/visualization/week5/assets/image-20240302102351083.png" referrerpolicy="no-referrer" alt="image-20240302102351083"></p><p><span>Классы в этой визуализации — это цифры, которые были представлены в датасете.</span></p><p><span>Благодаря этой визуализации мы можем понять как минимум несколько фактов:</span></p><ul><li><p><span>Классы «Один» и «Семь» находятся очень близко в пространстве. Это логично, поскольку цифры имеют похожее написание.</span></p></li><li><p><span>Модель плохо справилась с разведением основных классов, однако сильнее всего выделяет класс «Ноль» — точки на визуализации смешаны и толком не кластеризованы.</span></p></li><li><p><span>Очень близко в пространстве находятся классы «Три», «Шесть» и «Пять».</span></p></li></ul><p><span>Однако эта прикидка сделана «на глаз». Попробуем применить реальный алгоритм кластеризации и оценить результаты.</span></p><h2 id='k-means-clustering'><span>K-Means Clustering</span></h2><p><span>Создадим метод кластеризации K-Means. Посмотрим, насколько хорошо он справится. Количество классов поставим 10.</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python" style="break-inside: unset;"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation" style=""><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">cluster</span> <span class="cm-keyword">import</span> <span class="cm-variable">KMeans</span></span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">kmeans</span> <span class="cm-operator">=</span> <span class="cm-variable">KMeans</span>(<span class="cm-variable">n_clusters</span><span class="cm-operator">=</span><span class="cm-number">10</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X_clustered</span> <span class="cm-operator">=</span> <span class="cm-variable">kmeans</span>.<span class="cm-property">fit_predict</span>(<span class="cm-variable">X_5d</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-operator">/</span><span class="cm-variable">usr</span><span class="cm-operator">/</span><span class="cm-variable">local</span><span class="cm-operator">/</span><span class="cm-variable">lib</span><span class="cm-operator">/</span><span class="cm-variable">python3</span><span class="cm-number">.10</span><span class="cm-operator">/</span><span class="cm-variable">dist</span><span class="cm-operator">-</span><span class="cm-variable">packages</span><span class="cm-operator">/</span><span class="cm-variable">sklearn</span><span class="cm-operator">/</span><span class="cm-variable">cluster</span><span class="cm-operator">/</span><span class="cm-variable">_kmeans</span>.<span class="cm-property">py</span>:<span class="cm-number">870</span>: <span class="cm-variable">FutureWarning</span>: <span class="cm-variable">The</span> <span class="cm-variable">default</span> <span class="cm-variable">value</span> <span class="cm-variable">of</span> `<span class="cm-variable">n_init</span>` <span class="cm-variable">will</span> <span class="cm-variable">change</span> <span class="cm-keyword">from</span> <span class="cm-number">10</span> <span class="cm-variable">to</span> <span class="cm-string">'auto'</span> <span class="cm-keyword">in</span> <span class="cm-number">1.4</span>. <span class="cm-property">Set</span> <span class="cm-variable">the</span> <span class="cm-variable">value</span> <span class="cm-variable">of</span> `<span class="cm-variable">n_init</span>` <span class="cm-variable">explicitly</span> <span class="cm-variable">to</span> <span class="cm-variable">suppress</span> <span class="cm-variable">the</span> <span class="cm-variable">warning</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Create the scatter plot using matplotlib and seaborn</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">figure</span>(<span class="cm-variable">figsize</span><span class="cm-operator">=</span>(<span class="cm-number">10</span>, <span class="cm-number">8</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">sns</span>.<span class="cm-property">scatterplot</span>(</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">x</span><span class="cm-operator">=</span><span class="cm-variable">X_5d</span>[:, <span class="cm-number">0</span>],</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">y</span><span class="cm-operator">=</span><span class="cm-variable">X_5d</span>[:, <span class="cm-number">1</span>],</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">hue</span><span class="cm-operator">=</span><span class="cm-variable">X_clustered</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">palette</span><span class="cm-operator">=</span><span class="cm-string">'Accent'</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">legend</span><span class="cm-operator">=</span><span class="cm-keyword">False</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">edgecolor</span><span class="cm-operator">=</span><span class="cm-string">'white'</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">linewidth</span><span class="cm-operator">=</span><span class="cm-number">0.5</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">alpha</span><span class="cm-operator">=</span><span class="cm-number">0.8</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;">)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Set the title and labels</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">title</span>(<span class="cm-string">'KMeans Clustering'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">xlabel</span>(<span class="cm-string">'First Principal Component'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">ylabel</span>(<span class="cm-string">'Second Principal Component'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Show the legend</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">legend</span>(<span class="cm-variable">title</span><span class="cm-operator">=</span><span class="cm-string">'Cluster'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Show the plot</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">grid</span>(<span class="cm-keyword">True</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">show</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">WARNING</span>:<span class="cm-variable">matplotlib</span>.<span class="cm-property">legend</span>:<span class="cm-variable">No</span> <span class="cm-variable">artists</span> <span class="cm-keyword">with</span> <span class="cm-variable">labels</span> <span class="cm-variable">found</span> <span class="cm-variable">to</span> <span class="cm-variable">put</span> <span class="cm-keyword">in</span> <span class="cm-variable">legend</span>. &nbsp;<span class="cm-property">Note</span> <span class="cm-variable">that</span> <span class="cm-variable">artists</span> <span class="cm-variable">whose</span> <span class="cm-variable">label</span> <span class="cm-variable">start</span> <span class="cm-keyword">with</span> <span class="cm-variable">an</span> <span class="cm-variable">underscore</span> <span class="cm-variable">are</span> <span class="cm-variable">ignored</span> <span class="cm-variable">when</span> <span class="cm-variable">legend</span>() <span class="cm-keyword">is</span> <span class="cm-variable">called</span> <span class="cm-keyword">with</span> <span class="cm-variable">no</span> <span class="cm-variable">argument</span>.</span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 1092px;"></div><div class="CodeMirror-gutters" style="display: none; height: 1092px;"></div></div></div></pre><p><img src="/home/andrei/study/mipt_semester4/visualization/week5/assets/image-20240302102438397.png" referrerpolicy="no-referrer" alt="image-20240302102438397"></p><p><span>Уже зрительно мы видим, что наш алгоритм отработал плохо, — результаты сильно отличаются от оригинала. Однако какие-то общие паттерны все же проглядываются. Возможно, стоит использовать другой метод понижения размерности. Посмотрим, чем может помочь Linear Discriminant Analysis (LDA).</span></p><h2 id='linear-discriminant-analysis-lda'><span>Linear Discriminant Analysis (LDA)</span></h2><p><strong><span>LDA</span></strong><span> — очень популярный метод понижения размерности. Его цель — найти новые оси, называемые дискриминантами, которые максимизируют различия между классами данных. Он стремится максимизировать отношение разброса между классами к разбросу внутри классов.</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python" style="break-inside: unset;"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation" style=""><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">lda</span> <span class="cm-operator">=</span> <span class="cm-variable">LDA</span>(<span class="cm-variable">n_components</span><span class="cm-operator">=</span><span class="cm-number">5</span>)</span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Taking in as second argument the Target as labels</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X_LDA_2D</span> <span class="cm-operator">=</span> <span class="cm-variable">lda</span>.<span class="cm-property">fit_transform</span>(<span class="cm-variable">X_std</span>, <span class="cm-variable">Target</span>.<span class="cm-property">values</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">matplotlib</span>.<span class="cm-property">pyplot</span> <span class="cm-keyword">as</span> <span class="cm-variable">plt</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">seaborn</span> <span class="cm-keyword">as</span> <span class="cm-variable">sns</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Create the scatter plot using matplotlib and seaborn</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">figure</span>(<span class="cm-variable">figsize</span><span class="cm-operator">=</span>(<span class="cm-number">10</span>, <span class="cm-number">8</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">sns</span>.<span class="cm-property">scatterplot</span>(</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">x</span><span class="cm-operator">=</span><span class="cm-variable">X_LDA_2D</span>[:, <span class="cm-number">0</span>],</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">y</span><span class="cm-operator">=</span><span class="cm-variable">X_LDA_2D</span>[:, <span class="cm-number">1</span>],</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">hue</span><span class="cm-operator">=</span><span class="cm-variable">Target</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">palette</span><span class="cm-operator">=</span><span class="cm-string">'Accent'</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">legend</span><span class="cm-operator">=</span><span class="cm-keyword">True</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">edgecolor</span><span class="cm-operator">=</span><span class="cm-string">'white'</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">linewidth</span><span class="cm-operator">=</span><span class="cm-number">0.5</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">alpha</span><span class="cm-operator">=</span><span class="cm-number">0.8</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;">)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Set the title and labels</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">title</span>(<span class="cm-string">'Linear Discriminant Analysis (LDA)'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">xlabel</span>(<span class="cm-string">'First Linear Discriminant'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">ylabel</span>(<span class="cm-string">'Second Linear Discriminant'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Show the legend</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">legend</span>(<span class="cm-variable">title</span><span class="cm-operator">=</span><span class="cm-string">'Target'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Show the plot</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">grid</span>(<span class="cm-keyword">True</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">show</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 910px;"></div><div class="CodeMirror-gutters" style="display: none; height: 910px;"></div></div></div></pre><p><img src="/home/andrei/study/mipt_semester4/visualization/week5/assets/image-20240302102519364.png" referrerpolicy="no-referrer" alt="image-20240302102519364"></p><p><span>Уже лучше. Мы видим, что есть ряд классов, которые очень сильно отличаются друг от друга при понижении размерности. После LDA наши алгоритмы классификации сработали бы гораздо лучше.</span></p><p><span>Теперь посмотрим еще один метод понижения размерности — </span><strong><span>t-SNE</span></strong><span>.</span></p><h2 id='t-sne'><span>t-SNE</span></h2><p><strong><span>t-Distributed Stochastic Neighbor Embedding (t-SNE)</span></strong><span> — это метод машинного обучения для визуализации данных высокой размерности в пространство меньшей размерности, обычно двумерное или трехмерное. Он разработан для сохранения локальных структур данных, т. е. близость точек в исходном пространстве должна сохраняться в пространстве меньшей размерности.</span></p><p><span>Этот алгоритм идеально подходит для кластеризации и работы с алгоритмами после снижения размерности.</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python" style="break-inside: unset;"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation" style=""><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">tsne</span> <span class="cm-operator">=</span> <span class="cm-variable">TSNE</span>(<span class="cm-variable">n_components</span><span class="cm-operator">=</span><span class="cm-number">2</span>)</span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">tsne_results</span> <span class="cm-operator">=</span> <span class="cm-variable">tsne</span>.<span class="cm-property">fit_transform</span>(<span class="cm-variable">X_std</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">matplotlib</span>.<span class="cm-property">pyplot</span> <span class="cm-keyword">as</span> <span class="cm-variable">plt</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">seaborn</span> <span class="cm-keyword">as</span> <span class="cm-variable">sns</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Create the scatter plot using matplotlib and seaborn</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">figure</span>(<span class="cm-variable">figsize</span><span class="cm-operator">=</span>(<span class="cm-number">10</span>, <span class="cm-number">8</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">sns</span>.<span class="cm-property">scatterplot</span>(</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">x</span><span class="cm-operator">=</span><span class="cm-variable">tsne_results</span>[:, <span class="cm-number">0</span>],</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">y</span><span class="cm-operator">=</span><span class="cm-variable">tsne_results</span>[:, <span class="cm-number">1</span>],</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">hue</span><span class="cm-operator">=</span><span class="cm-variable">Target</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">palette</span><span class="cm-operator">=</span><span class="cm-string">'Accent'</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">legend</span><span class="cm-operator">=</span><span class="cm-keyword">True</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">edgecolor</span><span class="cm-operator">=</span><span class="cm-string">'white'</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">linewidth</span><span class="cm-operator">=</span><span class="cm-number">0.5</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">alpha</span><span class="cm-operator">=</span><span class="cm-number">0.8</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;">)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Set the title and labels</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">title</span>(<span class="cm-string">'TSNE (T-Distributed Stochastic Neighbour Embedding)'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">xlabel</span>(<span class="cm-string">'First Dimension'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">ylabel</span>(<span class="cm-string">'Second Dimension'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">legend</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Show the plot</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">grid</span>(<span class="cm-keyword">True</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">show</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 832px;"></div><div class="CodeMirror-gutters" style="display: none; height: 832px;"></div></div></div></pre><p><img src="/home/andrei/study/mipt_semester4/visualization/week5/assets/image-20240302102614175.png" referrerpolicy="no-referrer" alt="image-20240302102614175"></p><p><span>Здесь кластеры выделяются уже намного лучше. Но проблема в том, что в нашей палитре 8 цветов, и два цвета повторяются. Попробуем построить то же самое, но с использованием </span><code>colorbar</code><span>:</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python" style="break-inside: unset;"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation" style=""><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">train</span>[<span class="cm-string">'label'</span>] <span class="cm-operator">=</span> <span class="cm-variable">target</span></span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X</span> <span class="cm-operator">=</span> <span class="cm-variable">train</span>.<span class="cm-property">sample</span>(<span class="cm-variable">n</span><span class="cm-operator">=</span><span class="cm-number">10000</span>, <span class="cm-variable">random_state</span><span class="cm-operator">=</span><span class="cm-number">42</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">y</span> <span class="cm-operator">=</span> <span class="cm-variable">X</span>[<span class="cm-string">'label'</span>]</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X</span> <span class="cm-operator">=</span> <span class="cm-variable">X</span>.<span class="cm-property">drop</span>(<span class="cm-string">'label'</span>, <span class="cm-variable">axis</span> <span class="cm-operator">=</span> <span class="cm-number">1</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">tsne</span> <span class="cm-operator">=</span> <span class="cm-variable">TSNE</span>(<span class="cm-variable">n_components</span> <span class="cm-operator">=</span> <span class="cm-number">2</span>, <span class="cm-variable">random_state</span> <span class="cm-operator">=</span> <span class="cm-number">42</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X_reduced</span> <span class="cm-operator">=</span> <span class="cm-variable">tsne</span>.<span class="cm-property">fit_transform</span>(<span class="cm-variable">X</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">figure</span>(<span class="cm-variable">figsize</span><span class="cm-operator">=</span>(<span class="cm-number">12</span>, <span class="cm-number">8</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">tsne_results</span>[:,<span class="cm-number">0</span>], <span class="cm-variable">tsne_results</span>[:,<span class="cm-number">1</span>], <span class="cm-variable">c</span> <span class="cm-operator">=</span> <span class="cm-variable">Target</span>, <span class="cm-variable">cmap</span><span class="cm-operator">=</span><span class="cm-string">'jet'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">colorbar</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">axis</span>(<span class="cm-string">'off'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">show</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 390px;"></div><div class="CodeMirror-gutters" style="display: none; height: 390px;"></div></div></div></pre><p><img src="/home/andrei/study/mipt_semester4/visualization/week5/assets/image-20240302102647976.png" referrerpolicy="no-referrer" alt="image-20240302102647976"></p><p><span>Получившаяся ситуация очень похожа на реальную, которая наблюдается в данных. И это мы использовали всего лишь 10% от наших данных. Попробуйте самостоятельно выделить факты, которые можно понять о датасете и о написании цифр в целом, используя эту визуализацию.</span></p><h2 id='pca--t-sne'><span>PCA + t-SNE</span></h2><p><span>Стоит заметить, что разные методы понижения размерности можно совмещать, используя </span><code>Pipeline</code><span>:</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python" style="break-inside: unset;"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation" style=""><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">pipeline</span> <span class="cm-keyword">import</span> <span class="cm-variable">Pipeline</span></span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">pca_tsne</span> <span class="cm-operator">=</span> <span class="cm-variable">Pipeline</span>([</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;">(<span class="cm-string">'pca'</span>, <span class="cm-variable">PCA</span>(<span class="cm-variable">n_components</span><span class="cm-operator">=</span><span class="cm-number">95</span>, <span class="cm-variable">random_state</span><span class="cm-operator">=</span><span class="cm-number">42</span>)),</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;">(<span class="cm-string">'tsne'</span>, <span class="cm-variable">TSNE</span>(<span class="cm-variable">n_components</span><span class="cm-operator">=</span><span class="cm-number">2</span>, <span class="cm-variable">random_state</span><span class="cm-operator">=</span><span class="cm-number">42</span>)),</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;">])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X_pca_tsne</span> <span class="cm-operator">=</span> <span class="cm-variable">pca_tsne</span>.<span class="cm-property">fit_transform</span>(<span class="cm-variable">train</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">figure</span>(<span class="cm-variable">figsize</span><span class="cm-operator">=</span>(<span class="cm-number">12</span>, <span class="cm-number">8</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">X_reduced</span>[:,<span class="cm-number">0</span>], <span class="cm-variable">X_reduced</span>[:,<span class="cm-number">1</span>], <span class="cm-variable">c</span> <span class="cm-operator">=</span> <span class="cm-variable">target</span>, <span class="cm-variable">cmap</span><span class="cm-operator">=</span><span class="cm-string">'jet'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">colorbar</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">axis</span>(<span class="cm-string">'off'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">show</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 390px;"></div><div class="CodeMirror-gutters" style="display: none; height: 390px;"></div></div></div></pre><p><span>Однако данные в этом случае могут считаться очень долго. Например, этот код учится более часа. Но качество повышается.</span></p><h2 id='lle-locally-linear-embedding'><span>LLE (Locally Linear Embedding)</span></h2><p><span>Другой достаточно популярный алгоритм понижения размерности — LLE. Процесс LLE выполняется следующим образом:</span></p><ol start='' ><li><p><strong><span>Нахождение ближайших соседей.</span></strong><span> Сначала для каждой точки данных находятся ее ближайшие соседи в исходном пространстве.</span></p></li><li><p><strong><span>Поиск весовых коэффициентов.</span></strong><span> Затем LLE ищет линейную комбинацию соседей каждой точки, которая наилучшим образом воспроизводит эту точку. Это делается путем минимизации суммы квадратов разницы между точками и их линейными комбинациями.</span></p></li><li><p><strong><span>Представление в пространстве меньшей размерности.</span></strong><span> Наконец, LLE находит координаты точек в пространстве меньшей размерности таким образом, чтобы сохранить отношения близости между соседями, найденными на предыдущем шаге.</span></p></li></ol><p><span>Это специфически работающий алгоритм: он находит 𝑛-вершин, между которыми распределяет значения.</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation" style=""><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">manifold</span> <span class="cm-keyword">import</span> <span class="cm-variable">LocallyLinearEmbedding</span></span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X_lle</span> <span class="cm-operator">=</span> <span class="cm-variable">LocallyLinearEmbedding</span>(<span class="cm-variable">n_components</span><span class="cm-operator">=</span><span class="cm-number">2</span>, <span class="cm-variable">random_state</span><span class="cm-operator">=</span><span class="cm-number">42</span>).<span class="cm-property">fit_transform</span>(<span class="cm-variable">X</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">figure</span>(<span class="cm-variable">figsize</span><span class="cm-operator">=</span>(<span class="cm-number">12</span>, <span class="cm-number">8</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">X_lle</span>[:,<span class="cm-number">0</span>], <span class="cm-variable">X_lle</span>[:,<span class="cm-number">1</span>], <span class="cm-variable">c</span> <span class="cm-operator">=</span> <span class="cm-variable">y</span>, <span class="cm-variable">cmap</span><span class="cm-operator">=</span><span class="cm-string">'jet'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">colorbar</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">axis</span>(<span class="cm-string">'off'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">show</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 234px;"></div><div class="CodeMirror-gutters" style="display: none; height: 234px;"></div></div></div></pre><p><img src="/home/andrei/study/mipt_semester4/visualization/week5/assets/image-20240302102730481.png" referrerpolicy="no-referrer" alt="image-20240302102730481"></p><h1 id='применение-рассмотренного-подхода-на-примере'><span>Применение рассмотренного подхода на примере</span></h1><p><span>Возьмем датасет, который мы ранее использовали в работе на семинаре — датасет оценки качества яблок по ряду характеристик. Тогда мы строили визуализацию, используя только два столбца из всех доступных. Теперь попробуем применить разные методы понижения размерности для получения лучших результатов, при этом имея возможность использовать визуализацию.</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation"><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">apples</span> <span class="cm-operator">=</span> <span class="cm-variable">pd</span>.<span class="cm-property">read_csv</span>(<span class="cm-string">'https://raw.githubusercontent.com/ElijahSum/mipt_visualization/master/week_03_visualization/apple_quality.csv'</span>, <span class="cm-variable">index_col</span><span class="cm-operator">=</span><span class="cm-number">0</span>).<span class="cm-property">iloc</span>[:<span class="cm-operator">-</span><span class="cm-number">1</span>, :]</span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">apples</span></span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 104px;"></div><div class="CodeMirror-gutters" style="display: none; height: 104px;"></div></div></div></pre><figure class='table-figure'><table><thead><tr><th><span>A_id</span></th><th><strong><span>Size</span></strong></th><th><strong><span>Weight</span></strong></th><th><strong><span>Sweetness</span></strong></th><th><strong><span>Crunchiness</span></strong></th><th><strong><span>Juiciness</span></strong></th><th><strong><span>Ripeness</span></strong></th><th><strong><span>Acidity</span></strong></th><th><span>Quality</span></th></tr></thead><tbody><tr><td><strong><span>0.0</span></strong></td><td><span>-3.970049</span></td><td><span>-2.512336</span></td><td><span>5.346330</span></td><td><span>-1.012009</span></td><td><span>1.844900</span></td><td><span>0.329840</span></td><td><span>-0.491590483</span></td><td><span>good</span></td></tr><tr><td><strong><span>1.0</span></strong></td><td><span>-1.195217</span></td><td><span>-2.839257</span></td><td><span>3.664059</span></td><td><span>1.588232</span></td><td><span>0.853286</span></td><td><span>0.867530</span></td><td><span>-0.722809367</span></td><td><span>good</span></td></tr><tr><td><strong><span>2.0</span></strong></td><td><span>-0.292024</span></td><td><span>-1.351282</span></td><td><span>-1.738429</span></td><td><span>-0.342616</span></td><td><span>2.838636</span></td><td><span>-0.038033</span></td><td><span>2.621636473</span></td><td><span>bad</span></td></tr><tr><td><strong><span>3.0</span></strong></td><td><span>-0.657196</span></td><td><span>-2.271627</span></td><td><span>1.324874</span></td><td><span>-0.097875</span></td><td><span>3.637970</span></td><td><span>-3.413761</span></td><td><span>0.790723217</span></td><td><span>good</span></td></tr><tr><td><strong><span>4.0</span></strong></td><td><span>1.364217</span></td><td><span>-1.296612</span></td><td><span>-0.384658</span></td><td><span>-0.553006</span></td><td><span>3.030874</span></td><td><span>-1.303849</span></td><td><span>0.501984036</span></td><td><span>good</span></td></tr><tr><td><strong><span>...</span></strong></td><td><span>...</span></td><td><span>...</span></td><td><span>...</span></td><td><span>...</span></td><td><span>...</span></td><td><span>...</span></td><td><span>...</span></td><td><span>...</span></td></tr><tr><td><strong><span>3995.0</span></strong></td><td><span>0.059386</span></td><td><span>-1.067408</span></td><td><span>-3.714549</span></td><td><span>0.473052</span></td><td><span>1.697986</span></td><td><span>2.244055</span></td><td><span>0.137784369</span></td><td><span>bad</span></td></tr><tr><td><strong><span>3996.0</span></strong></td><td><span>-0.293118</span></td><td><span>1.949253</span></td><td><span>-0.204020</span></td><td><span>-0.640196</span></td><td><span>0.024523</span></td><td><span>-1.087900</span></td><td><span>1.854235285</span></td><td><span>good</span></td></tr><tr><td><strong><span>3997.0</span></strong></td><td><span>-2.634515</span></td><td><span>-2.138247</span></td><td><span>-2.440461</span></td><td><span>0.657223</span></td><td><span>2.199709</span></td><td><span>4.763859</span></td><td><span>-1.334611391</span></td><td><span>bad</span></td></tr><tr><td><strong><span>3998.0</span></strong></td><td><span>-4.008004</span></td><td><span>-1.779337</span></td><td><span>2.366397</span></td><td><span>-0.200329</span></td><td><span>2.161435</span></td><td><span>0.214488</span></td><td><span>-2.229719806</span></td><td><span>good</span></td></tr><tr><td><strong><span>3999.0</span></strong></td><td><span>0.278540</span></td><td><span>-1.715505</span></td><td><span>0.121217</span></td><td><span>-1.154075</span></td><td><span>1.266677</span></td><td><span>-0.776571</span></td><td><span>1.599796456</span></td><td><span>good</span></td></tr></tbody></table></figure><p><span>4000 rows × 8 columns</span></p><p><span>Создадим </span><code>test</code><span> и </span><code>train</code><span>, как делали ранее.</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation"><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">apples_train</span> <span class="cm-operator">=</span> <span class="cm-variable">apples</span>.<span class="cm-property">drop</span>(<span class="cm-string">'Quality'</span>, <span class="cm-variable">axis</span><span class="cm-operator">=</span><span class="cm-number">1</span>)</span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">apples_train</span>.<span class="cm-property">head</span>(<span class="cm-number">5</span>)</span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 52px;"></div><div class="CodeMirror-gutters" style="display: none; height: 52px;"></div></div></div></pre><figure class='table-figure'><table><thead><tr><th><span>A_id</span></th><th><strong><span>Size</span></strong></th><th><strong><span>Weight</span></strong></th><th><strong><span>Sweetness</span></strong></th><th><strong><span>Crunchiness</span></strong></th><th><strong><span>Juiciness</span></strong></th><th><strong><span>Ripeness</span></strong></th><th><strong><span>Acidity</span></strong></th></tr></thead><tbody><tr><td><strong><span>0.0</span></strong></td><td><span>-3.970049</span></td><td><span>-2.512336</span></td><td><span>5.346330</span></td><td><span>-1.012009</span></td><td><span>1.844900</span></td><td><span>0.329840</span></td><td><span>-0.491590483</span></td></tr><tr><td><strong><span>1.0</span></strong></td><td><span>-1.195217</span></td><td><span>-2.839257</span></td><td><span>3.664059</span></td><td><span>1.588232</span></td><td><span>0.853286</span></td><td><span>0.867530</span></td><td><span>-0.722809367</span></td></tr><tr><td><strong><span>2.0</span></strong></td><td><span>-0.292024</span></td><td><span>-1.351282</span></td><td><span>-1.738429</span></td><td><span>-0.342616</span></td><td><span>2.838636</span></td><td><span>-0.038033</span></td><td><span>2.621636473</span></td></tr><tr><td><strong><span>3.0</span></strong></td><td><span>-0.657196</span></td><td><span>-2.271627</span></td><td><span>1.324874</span></td><td><span>-0.097875</span></td><td><span>3.637970</span></td><td><span>-3.413761</span></td><td><span>0.790723217</span></td></tr><tr><td><strong><span>4.0</span></strong></td><td><span>1.364217</span></td><td><span>-1.296612</span></td><td><span>-0.384658</span></td><td><span>-0.553006</span></td><td><span>3.030874</span></td><td><span>-1.303849</span></td><td><span>0.501984036</span></td></tr></tbody></table></figure><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation"><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">apples_target</span> <span class="cm-operator">=</span> [<span class="cm-number">1</span> <span class="cm-keyword">if</span> <span class="cm-variable">x</span> <span class="cm-operator">==</span> <span class="cm-string">'good'</span> <span class="cm-keyword">else</span> <span class="cm-number">0</span> <span class="cm-keyword">for</span> <span class="cm-variable">x</span> <span class="cm-keyword">in</span> <span class="cm-variable">apples</span>[<span class="cm-string">'Quality'</span>]]</span></pre></div></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 26px;"></div><div class="CodeMirror-gutters" style="display: none; height: 26px;"></div></div></div></pre><p><span>Применим PCA-алгоритм и построим визуализацию полученного разбиения.</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python" style="break-inside: unset;"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation" style=""><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">pca</span> <span class="cm-operator">=</span> <span class="cm-variable">PCA</span>(<span class="cm-variable">n_components</span> <span class="cm-operator">=</span> <span class="cm-number">2</span>)</span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X_reduced</span> <span class="cm-operator">=</span> <span class="cm-variable">pca</span>.<span class="cm-property">fit_transform</span>(<span class="cm-variable">apples_train</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X_std</span> <span class="cm-operator">=</span> <span class="cm-variable">StandardScaler</span>().<span class="cm-property">fit_transform</span>(<span class="cm-variable">apples_train</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Call the PCA method with 95 components.</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">pca</span> <span class="cm-operator">=</span> <span class="cm-variable">PCA</span>(<span class="cm-variable">n_components</span><span class="cm-operator">=</span><span class="cm-number">5</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">pca</span>.<span class="cm-property">fit</span>(<span class="cm-variable">X_std</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X_5d</span> <span class="cm-operator">=</span> <span class="cm-variable">pca</span>.<span class="cm-property">transform</span>(<span class="cm-variable">X_std</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># For cluster coloring in our Plotly plots, remember to also restrict the target values</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">Target</span> <span class="cm-operator">=</span> <span class="cm-variable">target</span>[:<span class="cm-number">6000</span>]</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">figure</span>(<span class="cm-variable">figsize</span><span class="cm-operator">=</span>(<span class="cm-number">10</span>, <span class="cm-number">8</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">sns</span>.<span class="cm-property">scatterplot</span>(</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">x</span><span class="cm-operator">=</span><span class="cm-variable">X_5d</span>[:, <span class="cm-number">0</span>],</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">y</span><span class="cm-operator">=</span><span class="cm-variable">X_5d</span>[:, <span class="cm-number">1</span>],</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">hue</span><span class="cm-operator">=</span><span class="cm-variable">apples_target</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">palette</span><span class="cm-operator">=</span><span class="cm-string">'Accent'</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">legend</span><span class="cm-operator">=</span><span class="cm-string">'full'</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">edgecolor</span><span class="cm-operator">=</span><span class="cm-string">'white'</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">linewidth</span><span class="cm-operator">=</span><span class="cm-number">0.5</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">alpha</span><span class="cm-operator">=</span><span class="cm-number">0.8</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;">)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Set the title and labels</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">title</span>(<span class="cm-string">'Principal Component Analysis (PCA)'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">xlabel</span>(<span class="cm-string">'First Principal Component'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">ylabel</span>(<span class="cm-string">'Second Principal Component'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Show the legend</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">legend</span>(<span class="cm-variable">title</span><span class="cm-operator">=</span><span class="cm-string">'Target'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Show the plot</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">grid</span>(<span class="cm-keyword">True</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">show</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 1066px;"></div><div class="CodeMirror-gutters" style="display: none; height: 1066px;"></div></div></div></pre><p><img src="/home/andrei/study/mipt_semester4/visualization/week5/assets/image-20240302112824940.png" referrerpolicy="no-referrer" alt="image-20240302112824940"></p><p><span>Мы видим, что алгоритм сработал не очень хорошо. Попробуем другие доступные алгоритмы.</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python" style="break-inside: unset;"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation" style=""><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">tsne</span> <span class="cm-operator">=</span> <span class="cm-variable">TSNE</span>(<span class="cm-variable">n_components</span><span class="cm-operator">=</span><span class="cm-number">2</span>)</span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">tsne_results</span> <span class="cm-operator">=</span> <span class="cm-variable">tsne</span>.<span class="cm-property">fit_transform</span>(<span class="cm-variable">X_std</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Create the scatter plot using matplotlib and seaborn</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">figure</span>(<span class="cm-variable">figsize</span><span class="cm-operator">=</span>(<span class="cm-number">10</span>, <span class="cm-number">8</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">sns</span>.<span class="cm-property">scatterplot</span>(</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">x</span><span class="cm-operator">=</span><span class="cm-variable">tsne_results</span>[:, <span class="cm-number">0</span>],</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">y</span><span class="cm-operator">=</span><span class="cm-variable">tsne_results</span>[:, <span class="cm-number">1</span>],</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">hue</span><span class="cm-operator">=</span><span class="cm-variable">apples_target</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">palette</span><span class="cm-operator">=</span><span class="cm-string">'Accent'</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">legend</span><span class="cm-operator">=</span><span class="cm-keyword">True</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">edgecolor</span><span class="cm-operator">=</span><span class="cm-string">'white'</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">linewidth</span><span class="cm-operator">=</span><span class="cm-number">0.5</span>,</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">alpha</span><span class="cm-operator">=</span><span class="cm-number">0.8</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;">)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Set the title and labels</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">title</span>(<span class="cm-string">'TSNE (T-Distributed Stochastic Neighbour Embedding)'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">xlabel</span>(<span class="cm-string">'First Dimension'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">ylabel</span>(<span class="cm-string">'Second Dimension'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">legend</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Show the plot</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">grid</span>(<span class="cm-keyword">True</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">show</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 806px;"></div><div class="CodeMirror-gutters" style="display: none; height: 806px;"></div></div></div></pre><p><img src="/home/andrei/study/mipt_semester4/visualization/week5/assets/image-20240302112909068.png" referrerpolicy="no-referrer" alt="image-20240302112909068"></p><p><span>Мы видим, что алгоритм t-SNE сработал лучше, чем PCA. Здесь кластеры заметно разделены. Тут же можно сделать вывод: хорошие яблоки — это такие яблоки, которые сильно отходят от точки (0, 0) по обоим параметрам (разнонаправленно). Те яблоки, которые сильно отличаются только по оси x — плохие.</span></p><pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false" lang="python"><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang="python"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 11px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-code" role="presentation" style=""><div class="CodeMirror-activeline" style="position: relative;"><div class="CodeMirror-activeline-background CodeMirror-linebackground"></div><div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;"></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">X_lle</span> <span class="cm-operator">=</span> <span class="cm-variable">LocallyLinearEmbedding</span>(<span class="cm-variable">n_components</span><span class="cm-operator">=</span><span class="cm-number">2</span>, <span class="cm-variable">random_state</span><span class="cm-operator">=</span><span class="cm-number">42</span>).<span class="cm-property">fit_transform</span>(<span class="cm-variable">apples_train</span>)</span></pre></div><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="" cm-zwsp="">
</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">figure</span>(<span class="cm-variable">figsize</span><span class="cm-operator">=</span>(<span class="cm-number">12</span>, <span class="cm-number">8</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">X_lle</span>[:,<span class="cm-number">0</span>], <span class="cm-variable">X_lle</span>[:,<span class="cm-number">1</span>], <span class="cm-variable">c</span> <span class="cm-operator">=</span> <span class="cm-variable">apples_target</span>, <span class="cm-variable">cmap</span><span class="cm-operator">=</span><span class="cm-string">'jet'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">colorbar</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">axis</span>(<span class="cm-string">'off'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">plt</span>.<span class="cm-property">show</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 208px;"></div><div class="CodeMirror-gutters" style="display: none; height: 208px;"></div></div></div></pre><p><img src="/home/andrei/study/mipt_semester4/visualization/week5/assets/image-20240302112950222.png" referrerpolicy="no-referrer" alt="image-20240302112950222"></p><p><span>А этот алгоритм отработал хуже — точки находятся очень близко друг к другу.</span></p><h1 id='выводы'><span>Выводы</span></h1><ul><li><p><span>Методы понижения размерности можно использовать в двух сценариях: для ускорения работы алгоритма и для построения визуализаций, которые отражают закономерности в данных.</span></p></li><li><p><span>Самые распространённые методы понижения размерности: PCA, t-SNE, LDA и LLE.</span></p></li><li><p><span>Визуализация работы алгоритмов понижения размерности помогает проводить лучший EDA, потому показывает закономерности в данных.</span></p></li><li><p><span>Не все алгоритмы одинаково хорошо работают в разных датасетах: нужно экспериментировать с несколькими и искать те, которые смогут показать закономерности — кластеры, корреляции или любые другие зависимости в широком смысле слова. Если на визуализации каша — стоит попробовать другой алгоритм.</span></p></li><li><p><span>Если алгоритм отработал плохо, можно попробовать изменить гиперпараметры. Не существует универсального ответа на вопрос, сколько итераций нужно, чтобы разобраться, плохо отработал алгоритм или нужен тюнинг гиперпараметров.</span></p></li></ul></div></div>
</body>
</html>