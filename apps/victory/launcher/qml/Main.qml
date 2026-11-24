import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window
    width: 1920
    height: 1080
    minimumWidth: 1120
    minimumHeight: 700
    visible: true
    visibility: Window.Maximized
    title: "VIctory — Grand Theft Auto VI"
    color: "#0b0d10"
    font.family: "Segoe UI"

    property int pageIndex: 0
    readonly property color paper: "#f2eee8"
    readonly property color muted: "#a09b94"
    readonly property color accent: "#d83b4f"
    readonly property color iris: "#6257e8"
    readonly property color orchid: "#a445c4"
    readonly property color coral: "#e05c68"
    readonly property color sky: "#2b91ad"
    readonly property color surface: "#c715171b"
    readonly property color edge: "#2a2d32"

    Item {
        anchors.fill: parent
        z: -1
        Image {
            anchors.fill: parent
            source: launcher.backgroundSource
            fillMode: Image.PreserveAspectCrop
            sourceSize.width: 3840
            sourceSize.height: 2160
            asynchronous: true
            smooth: true
            opacity: 0.72
        }
        Rectangle {
            anchors.fill: parent
            gradient: Gradient {
                orientation: Gradient.Horizontal
                GradientStop { position: 0.0; color: "#bf090b10" }
                GradientStop { position: 0.48; color: "#8c090b10" }
                GradientStop { position: 1.0; color: "#5d090b10" }
            }
        }
        Rectangle {
            anchors.fill: parent
            gradient: Gradient {
                GradientStop { position: 0.0; color: "#20060910" }
                GradientStop { position: 1.0; color: "#7a060910" }
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        ColumnLayout {
            id: workspace
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 70
                gradient: Gradient {
                    orientation: Gradient.Horizontal
                    GradientStop { position: 0.0; color: "#eb10121b" }
                    GradientStop { position: 0.48; color: "#e2181421" }
                    GradientStop { position: 1.0; color: "#e00e1821" }
                }
                border.color: "#25282c"
                border.width: 1
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 24
                    anchors.rightMargin: 24
                    spacing: 14
                    RowLayout {
                        Layout.preferredWidth: 190
                        spacing: 10
                        Rectangle {
                            width: 36
                            height: 36
                            radius: 9
                            gradient: Gradient {
                                orientation: Gradient.Horizontal
                                GradientStop { position: 0.0; color: window.iris }
                                GradientStop { position: 1.0; color: window.coral }
                            }
                            border.color: "#8f798e"
                            border.width: 1
                            Text {
                                anchors.centerIn: parent
                                text: "VI"
                                color: "#f0d7c6"
                                font.family: "Georgia"
                                font.pixelSize: 16
                            }
                        }
                        ColumnLayout {
                            spacing: 1
                            Text {
                                text: "VICTORY"
                                color: window.paper
                                font.pixelSize: 12
                                font.weight: Font.Bold
                                font.letterSpacing: 1.5
                            }
                            Text {
                                text: "ROCKSTAR COLLECTION"
                                color: "#777b7e"
                                font.pixelSize: 8
                                font.weight: Font.DemiBold
                                font.letterSpacing: 0.7
                            }
                        }
                    }
                    Rectangle { width: 1; height: 30; color: "#303338" }
                    RowLayout {
                        spacing: 5
                        Repeater {
                            model: [
                                { label: "Library", page: 0 },
                                { label: "Activity", page: 1 },
                                { label: "Settings", page: 2 }
                            ]
                            delegate: Rectangle {
                                required property var modelData
                                Layout.preferredWidth: 94
                                Layout.preferredHeight: 40
                                radius: 7
                                gradient: Gradient {
                                    orientation: Gradient.Horizontal
                                    GradientStop { position: 0.0; color: window.pageIndex === modelData.page ? "#9a6257e8" : topNavMouse.containsMouse ? "#311f2534" : "transparent" }
                                    GradientStop { position: 1.0; color: window.pageIndex === modelData.page ? "#8ce05c68" : topNavMouse.containsMouse ? "#311f2534" : "transparent" }
                                }
                                Text {
                                    anchors.centerIn: parent
                                    text: modelData.label
                                    color: window.pageIndex === modelData.page ? "#f0e6e0" : "#a0a2a4"
                                    font.pixelSize: 12
                                    font.weight: window.pageIndex === modelData.page ? Font.DemiBold : Font.Normal
                                }
                                Rectangle {
                                    visible: window.pageIndex === modelData.page
                                    width: 24
                                    height: 2
                                    radius: 1
                                    color: window.accent
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    anchors.bottom: parent.bottom
                                }
                                MouseArea {
                                    id: topNavMouse
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: window.pageIndex = modelData.page
                                }
                            }
                        }
                    }
                    Item { Layout.fillWidth: true }
                    Rectangle {
                        Layout.preferredWidth: launcher.running ? 112 : 86
                        Layout.preferredHeight: 30
                        radius: 15
                        color: "#191b1e"
                        border.color: "#34373b"
                        border.width: 1
                        RowLayout {
                            anchors.centerIn: parent
                            spacing: 7
                            Rectangle {
                                width: 6
                                height: 6
                                radius: 3
                                color: launcher.running ? "#e2ac71" : "#7fb78a"
                            }
                            Text {
                                text: launcher.running ? "IN SESSION" : "LOCAL"
                                color: "#b9b8b4"
                                font.pixelSize: 9
                                font.weight: Font.Bold
                                font.letterSpacing: 1
                            }
                        }
                    }
                    Rectangle {
                        width: 34
                        height: 34
                        radius: 17
                        color: "#25272b"
                        border.color: "#45484c"
                        border.width: 1
                        Text {
                            anchors.centerIn: parent
                            text: "V"
                            color: "#e4c4ae"
                            font.family: "Georgia"
                            font.pixelSize: 15
                        }
                    }
                }
            }

            Item {
                id: desktopStage
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true

                StackLayout {
                    id: pages
                    anchors.fill: parent
                    currentIndex: window.pageIndex

                Item {
                    Flickable {
                        anchors.fill: parent
                        anchors.leftMargin: 28
                        anchors.rightMargin: 28
                        anchors.topMargin: 24
                        anchors.bottomMargin: 22
                        contentWidth: width
                        contentHeight: libraryContent.implicitHeight
                        clip: true
                        boundsBehavior: Flickable.StopAtBounds

                        ColumnLayout {
                            id: libraryContent
                            width: parent.width
                            spacing: 17

                            RowLayout {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 48
                                ColumnLayout {
                                    spacing: 2
                                    Text {
                                        text: "YOUR COLLECTION"
                                        color: "#86898c"
                                        font.pixelSize: 9
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.8
                                    }
                                    Text {
                                        text: "Game library"
                                        color: window.paper
                                        font.family: "Georgia"
                                        font.pixelSize: 25
                                    }
                                }
                                Item { Layout.fillWidth: true }
                                Text {
                                    text: "01  TITLE"
                                    color: "#85888c"
                                    font.pixelSize: 10
                                    font.weight: Font.DemiBold
                                    font.letterSpacing: 1.1
                                }
                            }

                            Rectangle {
                                id: featureCard
                                Layout.fillWidth: true
                                Layout.preferredHeight: 500
                                radius: 12
                                gradient: Gradient {
                                    orientation: Gradient.Horizontal
                                    GradientStop { position: 0.0; color: "#d3151b2c" }
                                    GradientStop { position: 1.0; color: "#e01b1620" }
                                }
                                border.color: "#5b414d"
                                border.width: 1
                                clip: true

                                RowLayout {
                                    anchors.fill: parent
                                    spacing: 0
                                    Item {
                                        Layout.fillWidth: true
                                        Layout.fillHeight: true
                                        Layout.preferredWidth: featureCard.width * 0.5
                                        Layout.minimumWidth: 300
                                        clip: true
                                        Image {
                                            anchors.fill: parent
                                            source: launcher.foregroundSource
                                            fillMode: Image.PreserveAspectCrop
                                            sourceSize.width: 3840
                                            sourceSize.height: 2160
                                            asynchronous: true
                                            smooth: true
                                        }
                                        Rectangle {
                                            anchors.fill: parent
                                            gradient: Gradient {
                                                orientation: Gradient.Horizontal
                                                GradientStop { position: 0.0; color: "#12090b10" }
                                                GradientStop { position: 0.66; color: "#3b090b10" }
                                                GradientStop { position: 1.0; color: "#b90a0c10" }
                                            }
                                        }
                                        Rectangle {
                                            anchors.left: parent.left
                                            anchors.top: parent.top
                                            anchors.leftMargin: 20
                                            anchors.topMargin: 20
                                            width: 144
                                            height: 29
                                            radius: 15
                                            color: "#bb101215"
                                            border.color: "#6e74777a"
                                            border.width: 1
                                            RowLayout {
                                                anchors.centerIn: parent
                                                spacing: 7
                                                Rectangle { width: 6; height: 6; radius: 3; color: window.accent }
                                                Text {
                                                    text: "ROCKSTAR GAMES"
                                                    color: "#e2ded9"
                                                    font.pixelSize: 9
                                                    font.weight: Font.Bold
                                                    font.letterSpacing: 0.9
                                                }
                                            }
                                        }
                                        ColumnLayout {
                                            anchors.left: parent.left
                                            anchors.bottom: parent.bottom
                                            anchors.leftMargin: 24
                                            anchors.bottomMargin: 23
                                            spacing: 5
                                            Text {
                                                text: "VICE CITY  ·  LEONIDA"
                                                color: "#f1c8ad"
                                                font.pixelSize: 10
                                                font.weight: Font.Bold
                                                font.letterSpacing: 1.6
                                            }
                                            Text {
                                                text: "A new chapter begins."
                                                color: "#f2eee9"
                                                font.family: "Georgia"
                                                font.pixelSize: 21
                                            }
                                        }
                                    }

                                    Rectangle {
                                        Layout.fillHeight: true
                                        Layout.fillWidth: true
                                        Layout.preferredWidth: featureCard.width * 0.5
                                        Layout.minimumWidth: 270
                                        gradient: Gradient {
                                            GradientStop { position: 0.0; color: "#e01a1725" }
                                            GradientStop { position: 1.0; color: "#e00d1a20" }
                                        }
                                        ColumnLayout {
                                            width: Math.min(parent.width - 72, 560)
                                            anchors.centerIn: parent
                                            spacing: 0
                                            RowLayout {
                                                Layout.fillWidth: true
                                                Rectangle {
                                                    Layout.preferredWidth: launcher.gameExists ? 76 : 90
                                                    Layout.preferredHeight: 25
                                                    radius: 13
                                                    color: launcher.gameExists ? "#1c3025" : "#332521"
                                                    border.color: launcher.gameExists ? "#365846" : "#614137"
                                                    border.width: 1
                                                    Text {
                                                        anchors.centerIn: parent
                                                        text: launcher.gameExists ? "READY" : "SETUP"
                                                        color: launcher.gameExists ? "#9dcca4" : "#e0ad8d"
                                                        font.pixelSize: 9
                                                        font.weight: Font.Bold
                                                        font.letterSpacing: 1
                                                    }
                                                }
                                                Item { Layout.fillWidth: true }
                                                Text {
                                                    text: "ROCKSTAR  /  VI"
                                                    color: "#85888b"
                                                    font.pixelSize: 9
                                                    font.weight: Font.DemiBold
                                                    font.letterSpacing: 0.9
                                                }
                                            }
                                            Text {
                                                Layout.fillWidth: true
                                                Layout.topMargin: 17
                                                text: "Grand Theft\nAuto VI"
                                                color: window.paper
                                                font.family: "Georgia"
                                                font.pixelSize: 34
                                                lineHeight: 0.92
                                            }
                                            Text {
                                                Layout.fillWidth: true
                                                Layout.topMargin: 9
                                                text: launcher.gameExists ? "Your game is ready. Start a session in " + launcher.consoleModel + "." : "Explore the state of Leonida with Jason and Lucia."
                                                color: "#a8a6a1"
                                                font.pixelSize: 12
                                                lineHeight: 1.25
                                                wrapMode: Text.WordWrap
                                            }
                                            Item { Layout.fillHeight: true }
                                            Text {
                                                Layout.fillWidth: true
                                                text: launcher.hasGame ? launcher.selectedPath : "Add the game executable from your existing install."
                                                color: "#86888a"
                                                font.pixelSize: 10
                                                elide: Text.ElideMiddle
                                            }
                                            Rectangle {
                                                Layout.fillWidth: true
                                                Layout.topMargin: 11
                                                Layout.preferredHeight: 51
                                                radius: 8
                                                gradient: Gradient {
                                                    orientation: Gradient.Horizontal
                                                    GradientStop { position: 0.0; color: playMouse.containsMouse ? "#8b6b62ef" : window.iris }
                                                    GradientStop { position: 0.52; color: playMouse.containsMouse ? "#a74f90df" : window.orchid }
                                                    GradientStop { position: 1.0; color: playMouse.containsMouse ? "#f27b72" : window.coral }
                                                }
                                                RowLayout {
                                                    anchors.fill: parent
                                                    anchors.leftMargin: 15
                                                    anchors.rightMargin: 15
                                                    spacing: 9
                                                    Text {
                                                        text: launcher.running ? "●" : "▶"
                                                        color: "white"
                                                        font.pixelSize: 11
                                                    }
                                                    Text {
                                                        Layout.fillWidth: true
                                                        horizontalAlignment: Text.AlignHCenter
                                                        text: launcher.running ? "SESSION IN PROGRESS" : launcher.gameExists ? "PLAY GRAND THEFT AUTO VI" : launcher.hasGame ? "LOCATE GAME FILES" : "ADD GAME FILES"
                                                        color: "white"
                                                        font.pixelSize: 11
                                                        font.weight: Font.Bold
                                                        font.letterSpacing: 0.8
                                                        elide: Text.ElideRight
                                                    }
                                                    Item { width: 12; height: 1 }
                                                }
                                                MouseArea {
                                                    id: playMouse
                                                    anchors.fill: parent
                                                    hoverEnabled: true
                                                    cursorShape: Qt.PointingHandCursor
                                                    enabled: !launcher.running
                                                    onClicked: launcher.gameExists ? launcher.launch() : launcher.browse()
                                                }
                                            }
                                            Rectangle {
                                                Layout.fillWidth: true
                                                Layout.topMargin: 8
                                                Layout.preferredHeight: 34
                                                radius: 7
                                                color: locateMouse.containsMouse ? "#22252a" : "transparent"
                                                Text {
                                                    anchors.centerIn: parent
                                                    text: launcher.hasGame ? "CHANGE GAME LOCATION" : "I'LL ADD FILES LATER"
                                                    color: "#aaa69f"
                                                    font.pixelSize: 9
                                                    font.weight: Font.DemiBold
                                                    font.letterSpacing: 0.8
                                                }
                                                MouseArea {
                                                    id: locateMouse
                                                    anchors.fill: parent
                                                    hoverEnabled: true
                                                    cursorShape: Qt.PointingHandCursor
                                                    onClicked: launcher.browse()
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 150
                                spacing: 12

                                Rectangle {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    radius: 12
                                    gradient: Gradient {
                                        GradientStop { position: 0.0; color: "#db211e30" }
                                        GradientStop { position: 1.0; color: "#de151b22" }
                                    }
                                    border.color: window.edge
                                    border.width: 1
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: 15
                                        spacing: 9
                                        Text {
                                            text: "GAME INSTALLATION"
                                            color: "#888b8f"
                                            font.pixelSize: 9
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.15
                                        }
                                        Text {
                                            Layout.fillWidth: true
                                            text: launcher.hasGame ? launcher.gameTitle : "Game files not added"
                                            color: window.paper
                                            font.family: "Georgia"
                                            font.pixelSize: 16
                                            elide: Text.ElideRight
                                        }
                                        Text {
                                            Layout.fillWidth: true
                                            Layout.fillHeight: true
                                            text: launcher.hasGame ? launcher.selectedPath : "Select the GTA VI executable in your existing folder."
                                            color: window.muted
                                            font.pixelSize: 10
                                            elide: Text.ElideMiddle
                                            wrapMode: Text.Wrap
                                        }
                                        Rectangle {
                                            Layout.preferredWidth: 122
                                            Layout.preferredHeight: 29
                                            radius: 6
                                            color: installMouse.containsMouse ? "#282b30" : "#202328"
                                            border.color: "#373b40"
                                            border.width: 1
                                            Text {
                                                anchors.centerIn: parent
                                                text: launcher.hasGame ? "CHANGE LOCATION" : "SELECT FILES"
                                                color: "#d9d3cb"
                                                font.pixelSize: 9
                                                font.weight: Font.DemiBold
                                                font.letterSpacing: 0.5
                                            }
                                            MouseArea {
                                                id: installMouse
                                                anchors.fill: parent
                                                hoverEnabled: true
                                                cursorShape: Qt.PointingHandCursor
                                                onClicked: launcher.browse()
                                            }
                                        }
                                    }
                                }

                                Rectangle {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    radius: 12
                                    gradient: Gradient {
                                        GradientStop { position: 0.0; color: "#db221a2d" }
                                        GradientStop { position: 1.0; color: "#de17191f" }
                                    }
                                    border.color: window.edge
                                    border.width: 1
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: 15
                                        spacing: 10
                                        Text {
                                            text: "CONSOLE PROFILE"
                                            color: "#888b8f"
                                            font.pixelSize: 9
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.15
                                        }
                                        Text {
                                            text: launcher.consoleModel
                                            color: window.paper
                                            font.family: "Georgia"
                                            font.pixelSize: 18
                                        }
                                        RowLayout {
                                            Layout.fillWidth: true
                                            Layout.fillHeight: true
                                            spacing: 6
                                            Repeater {
                                                model: ["Series X", "Series S"]
                                                delegate: Rectangle {
                                                    required property string modelData
                                                    Layout.fillWidth: true
                                                    Layout.preferredHeight: 32
                                                    radius: 6
                                                    gradient: Gradient {
                                                        orientation: Gradient.Horizontal
                                                        GradientStop { position: 0.0; color: launcher.consoleModel === modelData ? "#6242a1" : "#25232a" }
                                                        GradientStop { position: 1.0; color: launcher.consoleModel === modelData ? "#a84f76" : "#1b1d22" }
                                                    }
                                                    border.color: launcher.consoleModel === modelData ? "#81505c" : "#363a3f"
                                                    border.width: 1
                                                    Text {
                                                        anchors.centerIn: parent
                                                        text: modelData
                                                        color: launcher.consoleModel === modelData ? "#f0d6cf" : "#aaa9a5"
                                                        font.pixelSize: 10
                                                    }
                                                    MouseArea {
                                                        anchors.fill: parent
                                                        cursorShape: Qt.PointingHandCursor
                                                        onClicked: launcher.setConsoleModel(modelData)
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                Rectangle {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    radius: 12
                                    gradient: Gradient {
                                        GradientStop { position: 0.0; color: "#db1e202e" }
                                        GradientStop { position: 1.0; color: "#de171923" }
                                    }
                                    border.color: window.edge
                                    border.width: 1
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: 15
                                        spacing: 9
                                        Text {
                                            text: "RECENT ACTIVITY"
                                            color: "#888b8f"
                                            font.pixelSize: 9
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.15
                                        }
                                        Text {
                                            Layout.fillWidth: true
                                            Layout.fillHeight: true
                                            text: launcher.logs.length > 0 ? launcher.logs[launcher.logs.length - 1] : "Your local launch history will appear here."
                                            color: window.paper
                                            font.family: "Georgia"
                                            font.pixelSize: 14
                                            wrapMode: Text.WordWrap
                                            elide: Text.ElideRight
                                        }
                                        Rectangle {
                                            Layout.preferredWidth: 114
                                            Layout.preferredHeight: 29
                                            radius: 6
                                            color: activityMouse.containsMouse ? "#282b30" : "#202328"
                                            border.color: "#373b40"
                                            border.width: 1
                                            Text {
                                                anchors.centerIn: parent
                                                text: "VIEW ACTIVITY"
                                                color: "#d9d3cb"
                                                font.pixelSize: 9
                                                font.weight: Font.DemiBold
                                                font.letterSpacing: 0.5
                                            }
                                            MouseArea {
                                                id: activityMouse
                                                anchors.fill: parent
                                                hoverEnabled: true
                                                cursorShape: Qt.PointingHandCursor
                                                onClicked: window.pageIndex = 1
                                            }
                                        }
                                    }
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 24
                                Layout.topMargin: 1
                                Text {
                                    text: "QUICK ACCESS"
                                    color: "#c3bdc3"
                                    font.pixelSize: 9
                                    font.weight: Font.Bold
                                    font.letterSpacing: 1.6
                                }
                                Item { Layout.fillWidth: true }
                                Text {
                                    text: "LOCAL TOOLS  ·  THIS PC"
                                    color: "#85888e"
                                    font.pixelSize: 8
                                    font.weight: Font.DemiBold
                                    font.letterSpacing: 1
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 74
                                spacing: 10
                                Repeater {
                                    model: [
                                        { icon: "↻", label: "Run again", detail: "Last game", action: "run", c1: "#5140b7", c2: "#9c4eb8" },
                                        { icon: "⌖", label: "Check this PC", detail: "System details", action: "check", c1: "#295aaf", c2: "#5574d5" },
                                        { icon: "▤", label: "Open data folder", detail: "Saves & logs", action: "data", c1: "#246a83", c2: "#399d9d" },
                                        { icon: "⌂", label: "Set game folder", detail: "Install location", action: "folder", c1: "#73519f", c2: "#bb578f" },
                                        { icon: "⌨", label: "Keyboard controls", detail: "Play without pad", action: "controls", c1: "#883f83", c2: "#cf5675" },
                                        { icon: "◉", label: "Console model", detail: launcher.consoleModel, action: "console", c1: "#9a4c58", c2: "#dd6e66" },
                                        { icon: "×", label: "Quit launcher", detail: "Close VIctory", action: "quit", c1: "#563650", c2: "#984f71" }
                                    ]
                                    delegate: Rectangle {
                                        required property var modelData
                                        Layout.fillWidth: true
                                        Layout.fillHeight: true
                                        Layout.minimumWidth: 110
                                        radius: 10
                                        opacity: modelData.action === "run" && (!launcher.hasGame || !launcher.gameExists || launcher.running) ? 0.55 : 1.0
                                        gradient: Gradient {
                                            orientation: Gradient.Horizontal
                                            GradientStop { position: 0.0; color: actionMouse.containsMouse ? modelData.c1 : Qt.darker(modelData.c1, 1.24) }
                                            GradientStop { position: 1.0; color: actionMouse.containsMouse ? modelData.c2 : Qt.darker(modelData.c2, 1.32) }
                                        }
                                        border.color: actionMouse.containsMouse ? "#bfc2d8" : "#555360"
                                        border.width: 1
                                        RowLayout {
                                            anchors.fill: parent
                                            anchors.leftMargin: 12
                                            anchors.rightMargin: 9
                                            spacing: 9
                                            Text {
                                                text: modelData.icon
                                                color: "#f5eaf1"
                                                font.pixelSize: 20
                                                font.weight: Font.Light
                                            }
                                            ColumnLayout {
                                                Layout.fillWidth: true
                                                spacing: 3
                                                Text {
                                                    Layout.fillWidth: true
                                                    text: modelData.label
                                                    color: "#fff8f5"
                                                    font.pixelSize: 10
                                                    font.weight: Font.DemiBold
                                                    elide: Text.ElideRight
                                                }
                                                Text {
                                                    Layout.fillWidth: true
                                                    text: modelData.action === "console" ? launcher.consoleModel : modelData.detail
                                                    color: "#dfd5e2"
                                                    font.pixelSize: 8
                                                    elide: Text.ElideRight
                                                }
                                            }
                                        }
                                        MouseArea {
                                            id: actionMouse
                                            anchors.fill: parent
                                            hoverEnabled: true
                                            cursorShape: Qt.PointingHandCursor
                                            enabled: modelData.action !== "run" || (launcher.hasGame && launcher.gameExists && !launcher.running)
                                            onClicked: {
                                                if (modelData.action === "run") launcher.launch()
                                                else if (modelData.action === "check") window.pageIndex = 3
                                                else if (modelData.action === "data") launcher.openDataFolder()
                                                else if (modelData.action === "folder") launcher.setGameFolder()
                                                else if (modelData.action === "controls") controlsDialog.open()
                                                else if (modelData.action === "console") launcher.setConsoleModel(launcher.consoleModel === "Series X" ? "Series S" : "Series X")
                                                else if (modelData.action === "quit") window.close()
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Item {
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 30
                        spacing: 16
                        RowLayout {
                            Layout.fillWidth: true
                            ColumnLayout {
                                spacing: 3
                                Text {
                                    text: "ACTIVITY"
                                    color: "#85888b"
                                    font.pixelSize: 9
                                    font.weight: Font.Bold
                                    font.letterSpacing: 1.6
                                }
                                Text {
                                    text: "Launch history"
                                    color: window.paper
                                    font.family: "Georgia"
                                    font.pixelSize: 27
                                }
                            }
                            Item { Layout.fillWidth: true }
                            Rectangle {
                                Layout.preferredWidth: 130
                                Layout.preferredHeight: 34
                                radius: 7
                                color: clearMouse.containsMouse ? "#25282d" : "#191b1f"
                                border.color: "#383c41"
                                border.width: 1
                                Text {
                                    anchors.centerIn: parent
                                    text: "CLEAR HISTORY"
                                    color: "#d0cbc4"
                                    font.pixelSize: 9
                                    font.weight: Font.Bold
                                    font.letterSpacing: 0.8
                                }
                                MouseArea {
                                    id: clearMouse
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: launcher.clearActivity()
                                }
                            }
                        }
                        Text {
                            text: "GTA VI sessions and launcher messages stored on this PC."
                            color: window.muted
                            font.pixelSize: 12
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            radius: 12
                            gradient: Gradient {
                                GradientStop { position: 0.0; color: "#d91c2033" }
                                GradientStop { position: 1.0; color: "#d9171a20" }
                            }
                            border.color: window.edge
                            border.width: 1
                            ListView {
                                id: activityList
                                anchors.fill: parent
                                anchors.margins: 18
                                spacing: 10
                                clip: true
                                model: launcher.logs
                                onCountChanged: positionViewAtEnd()
                                delegate: Rectangle {
                                    required property string modelData
                                    required property int index
                                    width: activityList.width
                                    height: message.implicitHeight + 24
                                    radius: 8
                                    color: "#1c1f23"
                                    RowLayout {
                                        anchors.fill: parent
                                        anchors.margins: 12
                                        spacing: 10
                                        Rectangle {
                                            width: 6
                                            height: 6
                                            radius: 3
                                            color: index === activityList.count - 1 ? window.accent : "#74777b"
                                            Layout.alignment: Qt.AlignTop
                                            Layout.topMargin: 4
                                        }
                                        Text {
                                            id: message
                                            Layout.fillWidth: true
                                            text: modelData
                                            color: "#d3d0ca"
                                            font.family: "Cascadia Mono"
                                            font.pixelSize: 11
                                            wrapMode: Text.Wrap
                                        }
                                    }
                                }
                                Text {
                                    anchors.centerIn: parent
                                    visible: activityList.count === 0
                                    text: "No activity yet"
                                    color: "#92918c"
                                    font.family: "Georgia"
                                    font.pixelSize: 19
                                }
                            }
                        }
                    }
                }

                Item {
                    Flickable {
                        anchors.fill: parent
                        anchors.margins: 30
                        contentWidth: width
                        contentHeight: settingsContent.implicitHeight
                        clip: true
                        ColumnLayout {
                            id: settingsContent
                            width: parent.width
                            spacing: 14
                            Text {
                                text: "SETTINGS"
                                color: "#85888b"
                                font.pixelSize: 9
                                font.weight: Font.Bold
                                font.letterSpacing: 1.6
                            }
                            Text {
                                Layout.bottomMargin: 3
                                text: "Launcher preferences"
                                color: window.paper
                                font.family: "Georgia"
                                font.pixelSize: 27
                            }
                            Text {
                                Layout.bottomMargin: 5
                                text: "Manage your GTA VI profile, game files, and local data."
                                color: window.muted
                                font.pixelSize: 12
                            }
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 104
                                radius: 12
                                gradient: Gradient {
                                    GradientStop { position: 0.0; color: "#d8211b31" }
                                    GradientStop { position: 1.0; color: "#d71b1924" }
                                }
                                border.color: window.edge
                                border.width: 1
                                RowLayout {
                                    anchors.fill: parent
                                    anchors.margins: 18
                                    spacing: 13
                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        spacing: 4
                                        Text {
                                            text: "CONSOLE PROFILE"
                                            color: "#b88f88"
                                            font.pixelSize: 9
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.2
                                        }
                                        Text {
                                            text: "New sessions use " + launcher.consoleModel + "."
                                            color: window.paper
                                            font.family: "Georgia"
                                            font.pixelSize: 17
                                        }
                                    }
                                    Repeater {
                                        model: ["Series X", "Series S"]
                                        delegate: Rectangle {
                                            required property string modelData
                                            Layout.preferredWidth: 102
                                            Layout.preferredHeight: 34
                                            radius: 7
                                            gradient: Gradient {
                                                orientation: Gradient.Horizontal
                                                GradientStop { position: 0.0; color: launcher.consoleModel === modelData ? "#6242a1" : "#25232a" }
                                                GradientStop { position: 1.0; color: launcher.consoleModel === modelData ? "#a84f76" : "#1b1d22" }
                                            }
                                            border.color: launcher.consoleModel === modelData ? "#81505c" : "#363a3f"
                                            border.width: 1
                                            Text {
                                                anchors.centerIn: parent
                                                text: modelData
                                                color: launcher.consoleModel === modelData ? "#f0d6cf" : "#aaa9a5"
                                                font.pixelSize: 10
                                            }
                                            MouseArea {
                                                anchors.fill: parent
                                                cursorShape: Qt.PointingHandCursor
                                                onClicked: launcher.setConsoleModel(modelData)
                                            }
                                        }
                                    }
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 142
                                radius: 12
                                gradient: Gradient {
                                    GradientStop { position: 0.0; color: "#d71c2940" }
                                    GradientStop { position: 1.0; color: "#d9171a25" }
                                }
                                border.color: window.edge
                                border.width: 1
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 18
                                    spacing: 9
                                    Text {
                                        text: "GAME LOCATION"
                                        color: "#b88f88"
                                        font.pixelSize: 9
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.2
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: launcher.hasGame ? launcher.selectedPath : "Choose the GTA VI executable in your existing install folder."
                                        color: window.paper
                                        font.pixelSize: 11
                                        wrapMode: Text.WrapAnywhere
                                        maximumLineCount: 2
                                        elide: Text.ElideMiddle
                                    }
                                    Rectangle {
                                        Layout.preferredWidth: 164
                                        Layout.preferredHeight: 32
                                        radius: 7
                                        color: browseSettingsMouse.containsMouse ? "#282b30" : "#202328"
                                        border.color: "#373b40"
                                        border.width: 1
                                        Text {
                                            anchors.centerIn: parent
                                            text: launcher.hasGame ? "CHANGE LOCATION" : "LOCATE GTA VI"
                                            color: "#ddd6cf"
                                            font.pixelSize: 9
                                            font.weight: Font.DemiBold
                                            font.letterSpacing: 0.7
                                        }
                                        MouseArea {
                                            id: browseSettingsMouse
                                            anchors.fill: parent
                                            hoverEnabled: true
                                            cursorShape: Qt.PointingHandCursor
                                            onClicked: launcher.browse()
                                        }
                                    }
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 115
                                radius: 12
                                gradient: Gradient {
                                    GradientStop { position: 0.0; color: "#d824202b" }
                                    GradientStop { position: 1.0; color: "#d7191922" }
                                }
                                border.color: window.edge
                                border.width: 1
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 18
                                    spacing: 8
                                    Text {
                                        text: "LOCAL DATA"
                                        color: "#b88f88"
                                        font.pixelSize: 9
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.2
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: launcher.dataDirectory
                                        color: window.paper
                                        font.family: "Cascadia Mono"
                                        font.pixelSize: 10
                                        wrapMode: Text.Wrap
                                    }
                                    Text {
                                        text: "Preferences and launch activity stay on this PC."
                                        color: window.muted
                                        font.pixelSize: 10
                                    }
                                }
                            }
                        }
                    }
                }

                Item {
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 34
                        spacing: 18
                        RowLayout {
                            Layout.fillWidth: true
                            ColumnLayout {
                                spacing: 4
                                Text {
                                    text: "SYSTEM PROFILE  ·  LOCAL DIAGNOSTICS"
                                    color: "#a99dd1"
                                    font.pixelSize: 9
                                    font.weight: Font.Bold
                                    font.letterSpacing: 1.6
                                }
                                Text {
                                    text: "This PC"
                                    color: window.paper
                                    font.family: "Georgia"
                                    font.pixelSize: 30
                                }
                            }
                            Item { Layout.fillWidth: true }
                            Rectangle {
                                Layout.preferredWidth: 132
                                Layout.preferredHeight: 38
                                radius: 9
                                gradient: Gradient {
                                    orientation: Gradient.Horizontal
                                    GradientStop { position: 0.0; color: "#634eae" }
                                    GradientStop { position: 1.0; color: "#ac527e" }
                                }
                                Text {
                                    anchors.centerIn: parent
                                    text: "BACK TO LIBRARY"
                                    color: "#fff7f3"
                                    font.pixelSize: 9
                                    font.weight: Font.Bold
                                    font.letterSpacing: 0.7
                                }
                                MouseArea {
                                    anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: window.pageIndex = 0
                                }
                            }
                        }
                        Text {
                            Layout.fillWidth: true
                            text: "A quick readout of Windows, processor, graphics adapter, and memory. Rendering feature level is confirmed by the game when it starts."
                            color: window.muted
                            font.pixelSize: 12
                            wrapMode: Text.WordWrap
                        }
                        GridLayout {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            columns: 2
                            rowSpacing: 12
                            columnSpacing: 12
                            Repeater {
                                model: launcher.systemInfo
                                delegate: Rectangle {
                                    required property var modelData
                                    required property int index
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    Layout.minimumHeight: 145
                                    radius: 13
                                    gradient: Gradient {
                                        orientation: Gradient.Horizontal
                                        GradientStop { position: 0.0; color: ["#29234a", "#202b4b", "#242548", "#3a2446"][index] }
                                        GradientStop { position: 1.0; color: ["#211b34", "#1c273b", "#1c2e38", "#30202e"][index] }
                                    }
                                    border.color: ["#53427b", "#3e5380", "#3b6973", "#77415e"][index]
                                    border.width: 1
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: 22
                                        spacing: 8
                                        RowLayout {
                                            Layout.fillWidth: true
                                            Text {
                                                text: modelData.label
                                                color: "#bdb5d5"
                                                font.pixelSize: 9
                                                font.weight: Font.Bold
                                                font.letterSpacing: 1.3
                                            }
                                            Item { Layout.fillWidth: true }
                                            Text {
                                                text: modelData.status
                                                color: modelData.status === "READY" ? "#9be3bd" : modelData.status === "INFO" ? "#a6dce5" : "#efc38e"
                                                font.pixelSize: 8
                                                font.weight: Font.Bold
                                                font.letterSpacing: 0.8
                                            }
                                        }
                                        Text {
                                            Layout.fillWidth: true
                                            text: modelData.value
                                            color: window.paper
                                            font.family: "Georgia"
                                            font.pixelSize: 19
                                            maximumLineCount: 2
                                            elide: Text.ElideRight
                                            wrapMode: Text.WordWrap
                                        }
                                        Item { Layout.fillHeight: true }
                                        Text {
                                            Layout.fillWidth: true
                                            text: modelData.detail
                                            color: "#b4b0ba"
                                            font.pixelSize: 10
                                            wrapMode: Text.WordWrap
                                        }
                                    }
                                }
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10
                            Text {
                                Layout.fillWidth: true
                                text: "Game folder: " + (launcher.gameDirectory || "Not set")
                                color: "#aaa4b0"
                                font.pixelSize: 10
                                elide: Text.ElideMiddle
                            }
                            Rectangle {
                                Layout.preferredWidth: 152
                                Layout.preferredHeight: 36
                                radius: 8
                                color: "#25304a"
                                border.color: "#485c87"
                                border.width: 1
                                Text { anchors.centerIn: parent; text: "SET GAME FOLDER"; color: "#e8e1f1"; font.pixelSize: 9; font.weight: Font.DemiBold }
                                MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: launcher.setGameFolder() }
                            }
                            Rectangle {
                                Layout.preferredWidth: 144
                                Layout.preferredHeight: 36
                                radius: 8
                                color: "#3d2444"
                                border.color: "#75486f"
                                border.width: 1
                                Text { anchors.centerIn: parent; text: "OPEN DATA FOLDER"; color: "#f0e0ee"; font.pixelSize: 9; font.weight: Font.DemiBold }
                                MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: launcher.openDataFolder() }
                            }
                        }
                    }
                }
            }
            }
        }
    }

    Dialog {
        id: controlsDialog
        modal: true
        anchors.centerIn: Overlay.overlay
        width: Math.min(window.width - 40, 780)
        height: Math.min(window.height - 40, 560)
        padding: 0
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
        background: Rectangle {
            radius: 18
            gradient: Gradient {
                orientation: Gradient.Horizontal
                GradientStop { position: 0.0; color: "#f0211b35" }
                GradientStop { position: 0.55; color: "#f01d2137" }
                GradientStop { position: 1.0; color: "#f02d1c31" }
            }
            border.color: "#80576f"
            border.width: 1
        }
        contentItem: ColumnLayout {
            anchors.fill: parent
            anchors.margins: 26
            spacing: 15
            RowLayout {
                Layout.fillWidth: true
                ColumnLayout {
                    spacing: 4
                    Text { text: "KEYBOARD CONTROLS"; color: "#d1a8e0"; font.pixelSize: 9; font.weight: Font.Bold; font.letterSpacing: 1.6 }
                    Text { text: "Play your way"; color: window.paper; font.family: "Georgia"; font.pixelSize: 26 }
                }
                Item { Layout.fillWidth: true }
                Rectangle {
                    width: 36; height: 36; radius: 18
                    color: "#3b2b47"; border.color: "#775279"; border.width: 1
                    Text { anchors.centerIn: parent; text: "×"; color: "#f4e6f0"; font.pixelSize: 19 }
                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: controlsDialog.close() }
                }
            }
            Text {
                Layout.fillWidth: true
                text: "Use these keyboard bindings as a fallback when a controller is not connected."
                color: "#b7afbd"
                font.pixelSize: 11
                wrapMode: Text.WordWrap
            }
            GridLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                columns: 2
                rowSpacing: 9
                columnSpacing: 9
                Repeater {
                    model: [
                        { key: "W A S D  /  I J K L", label: "Left and right sticks" },
                        { key: "ARROW KEYS", label: "D-pad" },
                        { key: "ENTER", label: "A button" },
                        { key: "BACKSPACE", label: "B button" },
                        { key: "X  /  Y", label: "X and Y buttons" },
                        { key: "Q  /  E", label: "Left and right shoulders" },
                        { key: "Z  /  C", label: "Left and right triggers" },
                        { key: "F  /  H", label: "Stick clicks" },
                        { key: "ESC  /  TAB", label: "Menu and View" }
                    ]
                    delegate: Rectangle {
                        required property var modelData
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.minimumHeight: 66
                        radius: 10
                        gradient: Gradient {
                            orientation: Gradient.Horizontal
                            GradientStop { position: 0.0; color: "#7b392d3e" }
                            GradientStop { position: 1.0; color: "#68312f47" }
                        }
                        border.color: "#654966"
                        border.width: 1
                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 15
                            anchors.rightMargin: 15
                            spacing: 12
                            Text {
                                text: modelData.key
                                color: "#f1d1e4"
                                font.family: "Cascadia Mono"
                                font.pixelSize: 10
                                font.weight: Font.Bold
                            }
                            Rectangle { width: 1; height: 28; color: "#705270" }
                            Text {
                                Layout.fillWidth: true
                                text: modelData.label
                                color: "#e3dce3"
                                font.pixelSize: 11
                                wrapMode: Text.WordWrap
                            }
                        }
                    }
                }
            }
        }
    }
}
